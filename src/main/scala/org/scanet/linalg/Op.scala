package org.scanet.linalg

import java.util.UUID

import org.scanet.core.Numeric
import org.scanet.linalg.Op.BuilderState._
import org.scanet.linalg.Op._
import org.tensorflow.{OperationBuilder, Output}
import org.tensorflow.op.{Scope => NativeScope}

case class Op[A: Numeric](
       name: String,
       label: String,
       shape: Shape,
       inputs: List[Op[A]],
       compiler: OpContext[A] => Output[A]) {

  val id: String = UUID.randomUUID().toString

  def rank: Int = shape.rank

  def compile(context: Context): (Context, Compiled[A]) = {
    val (contextAfterInput, outputs) = inputs.foldLeft((context, List[Output[A]]()))(
      (acc, op) => {
        val (currentContext, outs) = acc
        val (newContext, out) = op.findOrCompile(currentContext)
        (newContext, out._2::outs)
      })
    val uniqueLabel = Label(label, contextAfterInput.maxLabelIndex(label) + 1)
    val output = compiler(OpContext(contextAfterInput, this, uniqueLabel, outputs.reverse))
    val compiled = (uniqueLabel, output)
    val newCache = contextAfterInput.outputs + (id -> compiled.asInstanceOf[Compiled[AnyRef]])
    val contextAfterOutput = contextAfterInput.copy(outputs = newCache)
    (contextAfterOutput, compiled)
  }

  def findOrCompile(context: Context): (Context, Compiled[A]) = {
    context.outputs.get(id)
      .map(compiled => (context, compiled.asInstanceOf[Compiled[A]]))
      .getOrElse {compile(context)}
  }

  def upstreamOptions: List[Op[A]] = {
    inputs.flatMap(op => op.upstreamOptions)
  }

  override def toString: String = {
    // todo: add shapes
    val args = if (inputs.nonEmpty) s"(${inputs.mkString(", ")})" else ""
    val fullName = if (label == name) s"$name" else s"$label:$name"
    s"$fullName$args:$shape"
  }

  def eval: Tensor[A] = Session.run(this)
}

object Op {

  case class Label(value: String, index: Int = 0) {
    require(value.nonEmpty, "name cannot be empty")
    require(index >= -1, "index should be positive or -1")
    override def toString: String = s"${value}_$index"
  }

  type Compiled[A] = (Label, Output[A])

  case class OpContext[A: Numeric](global: Context, op: Op[A], label: Label, inputs: List[Output[A]])

  case class Context(scope: NativeScope, outputs: Map[String, Compiled[AnyRef]]) {
    def maxLabelIndex(name: String): Int = {
      // NOTE: in the future think about prebuilt index
      val names = outputs.values.map(_._1).groupBy(_.value)
      names.get(name).map(n => n.map(_.index).max).getOrElse(-1)
    }
  }

  sealed trait BuilderState
  object BuilderState {
    sealed trait WithName extends BuilderState
    sealed trait WithLabel extends BuilderState
    sealed trait WithShape extends BuilderState
    sealed trait WithCompiler extends BuilderState
    type Complete = WithName with WithLabel with WithShape with WithCompiler
    type Transformer[A] = (List[Output[A]], OperationBuilder) => OperationBuilder
  }


  case class Builder[A: Numeric, State <: BuilderState](
        name: String,
        label: String,
        shape: Shape,
        inputs: List[Op[A]],
        transformers: List[Transformer[A]]) {

    def label(label: String): Builder[A, State with WithLabel] = copy(label = label)

    def shape(shape: Shape): Builder[A, State with WithShape] = copy(shape = shape)

    def inputs(inputs: Op[A]*): Builder[A, State] = copy(inputs = inputs.toList)

    def compileWithTransformer(f: Transformer[A]): Builder[A, State with WithCompiler] =
      copy(transformers = f :: transformers)

    def compileWithValue(tensor: Tensor[A]): Builder[A, State with WithCompiler] =
      compileWithTransformer((_, builder) => builder
        .setAttr("value", tensor)
        .setAttr("dtype", Numeric[A].tag)
      )

    def compileWithAllInputs: Builder[A, State with WithCompiler] =
      compileWithTransformer((inputs, builder) =>
        inputs.foldLeft(builder)((acc, next) => acc.addInput(next)))

    def build(implicit ev: State =:= Complete): Op[A] = {
      Op[A](name, label, shape, inputs, (context: OpContext[A]) => {
        val init = context.global.scope.env.opBuilder(context.op.name, context.label.toString)
        val transformed = transformers.foldLeft(init)((acc, next) => next(context.inputs, acc))
        transformed.build().output(0).asInstanceOf[Output[A]]
      })
    }
  }

  def name[A: Numeric](name: String): Builder[A, WithName] = {
    Builder[A, WithName](name, label = null, shape = null, inputs = Nil, transformers = Nil)
  }

  def const[A: Numeric](value: A): Op[A] =
    const(Tensor.scalar[A](value))

  def const[A: Numeric](value: A, name: String): Op[A] =
    const(Tensor.scalar[A](value), name)

  def const[A: Numeric](tensor: Tensor[A]): Op[A] = const(tensor, "Const")

  def const[A: Numeric](tensor: Tensor[A], label: String): Op[A] =
     Op.name[A]("Const")
       .label(label)
       .shape(tensor.shape)
       .compileWithValue(tensor)
       .build

  def plus[A: Numeric](left: Op[A], right: Op[A], label: String = "Add"): Op[A] = {
    require(left.shape.endsWith(right.shape) || right.shape.endsWith(left.shape) ,
      s"tensors with shapes ${left.shape} and ${right.shape} cannot be added, " +
        "one of the tensors should have shape which includes the other")
    Op.name("Add")
      .label(label)
      .shape(if (left.rank > right.rank) left.shape else right.shape)
      .inputs(left, right)
      .compileWithAllInputs
      .build
  }
}

