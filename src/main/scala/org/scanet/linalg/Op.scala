package org.scanet.linalg

import java.util.UUID

import org.scanet.core.Numeric
import org.scanet.linalg.Op.BuilderState._
import org.tensorflow.{OperationBuilder, Output}
import org.tensorflow.op.{Scope => NativeScope}

case class Context(scope: NativeScope, cache: Map[String, Output[AnyRef]])

case class Label(value: String, index: Int = 0) {
  require(value.nonEmpty, "name cannot be empty")
  require(index >= -1, "index should be positive or -1")

  override def toString: String = s"${value}_$index"
}

case class Op[A: Numeric](
       name: String,
       label: Label,
       shape: Shape,
       inputs: List[Op[A]],
       compiler: (Op[A], Context, List[Output[A]]) => Output[A]) {

  val id: String = UUID.randomUUID().toString

  def rank: Int = shape.rank

  def compile(context: Context): (Context, Output[A]) = {
    val (context1, outputs) = inputs.foldLeft((context, List[Output[A]]()))(
      (acc, op) => {
        val (currentContext, outs) = acc
        val (newContext, out) = op.findOrCompile(currentContext)
        (newContext, out::outs)
      })
    val output = compiler(this, context1, outputs.reverse)
    val newCache = context1.cache + (id -> output.asInstanceOf[Output[AnyRef]])
    val context2 = context1.copy(cache = newCache)
    (context2, output)
  }

  def findOrCompile(context: Context): (Context, Output[A]) = {
    context.cache.get(id)
      .map(operand => (context, operand.asInstanceOf[Output[A]]))
      .getOrElse {compile(context)}
  }

  def upstreamOptions: List[Op[A]] = {
    inputs.flatMap(op => op.upstreamOptions)
  }

  def maxIndexOfName(name: String): Int = {
    // NOTE: in the future think about prebuilt index
    val names = upstreamOptions.map(_.label).groupBy(_.value)
    names.get(name).map(n => n.map(_.index).max).getOrElse(-1)
  }

  def withUniqueName: Op[A] = {
    val maxIndex = maxIndexOfName(label.value)
    if (maxIndex >= label.index) {
      copy(label = label.copy(index = maxIndex + 1))
    } else {
      this
    }
  }

  override def toString: String = {
    val args = if (inputs.nonEmpty) s"(${inputs.mkString(", ")})" else ""
    s"$label:$name$args"
  }

  def eval: Tensor[A] = Session.run(this)
}

object Op {

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
      Op[A](name, label, shape, inputs, (op: Op[A], context, inputs: List[Output[A]]) => {
        val init = context.scope.env.opBuilder(op.name, op.label.toString)
        val transformed = transformers.foldLeft(init)((acc, next) => next(inputs, acc))
        transformed.build().output(0).asInstanceOf[Output[A]]
      })
    }
  }

  def name[A: Numeric](name: String): Builder[A, WithName] = {
    Builder[A, WithName](name, label = null, shape = null, inputs = Nil, transformers = Nil)
  }

  def apply[A: Numeric](
       name: String,
       label: String,
       shape: Shape,
       inputs: List[Op[A]],
       compiler: (Op[A], Context, List[Output[A]]) => Output[A]): Op[A] = {
    // todo: fix unique names - make unique names before evaluation
    new Op(name, Label(label), shape, inputs, compiler).withUniqueName
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
    // todo: lower dimensions should be equal and tensor with higher dimension
    require(left.shape.last == right.shape.last || left.shape.isScalar || right.shape.isScalar,
      s"tensors with shapes ${left.shape} and ${right.shape} cannot be added, " +
        "either last dimensions should equal or one of the tensors should be a scalar")
    Op.name("Add")
      .label(label)
      .shape(if (left.rank > right.rank) left.shape else right.shape)
      .inputs(left, right)
      .compileWithAllInputs
      .build
  }
}

