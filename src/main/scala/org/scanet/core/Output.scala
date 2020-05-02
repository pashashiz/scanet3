package org.scanet.core

import java.util.UUID

import org.scanet.core
import org.scanet.core.Output.BuilderState._
import org.scanet.core.Output._
import org.tensorflow.op.{Scope => NativeScope}
import org.tensorflow.{Operation, OperationBuilder}

case class Output[A: TensorType](
              name: String,
              label: String,
              shape: Shape,
              inputs: List[Output[A]],
              controls: List[Output[A]],
              compiler: OpContext[A] => Operation) {

  val id: String = UUID.randomUUID().toString

  def rank: Int = shape.rank

  def broadcastableBy(smaller: Output[A]): Boolean = shape.broadcastableBy(smaller.shape)

  def broadcastableAny(other: Output[A]): Boolean = shape.broadcastableAny(other.shape)

  def compile(context: Context): (Context, Compiled[A]) = {
    val (contextAfterInput, outputs) = compileInputs(inputs, context)
    val (contextAfterControls, controlOuts) = compileInputs(controls, contextAfterInput)
    val uniqueLabel = Label(label, contextAfterControls.maxLabelIndex(label) + 1)
    val output = compiler(OpContext(contextAfterControls, this, uniqueLabel, outputs.reverse, controlOuts))
    val compiled = (uniqueLabel, output)
    val newCache = contextAfterControls.outputs + (id -> compiled)
    val contextAfterOutput = contextAfterControls.copy(outputs = newCache)
    (contextAfterOutput, compiled)
  }

  private def compileInputs(in: List[Output[A]], context: Context): (Context, List[Operation]) = {
    val (contextAfterInput, outputs) = in.foldLeft((context, List[Operation]()))(
      (acc, op) => {
        val (currentContext, outs) = acc
        val (newContext, out) = op.findOrCompile(currentContext)
        (newContext, out._2 :: outs)
      })
    (contextAfterInput, outputs)
  }

  def findOrCompile(context: Context): (Context, Compiled[A]) = {
    context.outputs.get(id)
      .map(compiled => (context, compiled))
      .getOrElse {compile(context)}
  }

  def upstreamOptions: List[Output[A]] = {
    inputs.flatMap(op => op.upstreamOptions)
  }

  override def toString: String = {
    val args = if (inputs.nonEmpty) s"(${inputs.mkString(", ")})" else ""
    val fullName = if (label == name) s"$name" else s"$label:$name"
    s"$fullName$args:$shape"
  }
}

object Output {

  case class Label(value: String, index: Int = 0) {
    require(value.nonEmpty, "name cannot be empty")
    require(index >= -1, "index should be positive or -1")
    override def toString: String = s"${value}_$index"
  }

  type Compiled[A] = (Label, Operation)

  case class OpContext[A: TensorType](
      global: Context,
      op: Output[A],
      label: Label,
      inputs: List[Operation],
      controls: List[Operation])

  case class Context(scope: NativeScope, outputs: Map[String, Compiled[_]]) {
    def maxLabelIndex(name: String): Int = {
      // NOTE: in the future think about prebuilt index
      val names = outputs.values.map(_._1).groupBy(_.value)
      names.get(name).map(n => n.map(_.index).max).getOrElse(-1)
    }
  }

  sealed trait BuilderState
  object BuilderState {
    sealed trait WithName extends BuilderState
    sealed trait WithShape extends BuilderState
    sealed trait WithCompiler extends BuilderState
    type Complete = WithName with WithShape with WithCompiler
    type Transformer[A] = (OpContext[A], OperationBuilder) => OperationBuilder
  }

  case class Builder[A: TensorType, State <: BuilderState](
        name: String,
        label: String,
        shape: Shape,
        inputs: List[Output[A]],
        controls: List[Output[A]],
        transformers: List[Transformer[A]]) {

    def label(label: String): Builder[A, State] = copy(label = label)

    def shape(shape: Shape): Builder[A, State with WithShape] = copy(shape = shape)

    def inputs(inputs: Output[_]*): Builder[A, State] = {
      copy(inputs = inputs.toList.asInstanceOf[List[Output[A]]])
    }

    def controlInputs(controls: Output[_]*): Builder[A, State] = {
      copy(controls = controls.toList.asInstanceOf[List[Output[A]]])
    }

    def compileWithTransformer(f: Transformer[A]): Builder[A, State with WithCompiler] =
      copy(transformers = f :: transformers)

    def compileWithValue(tensor: Tensor[A]): Builder[A, State with WithCompiler] =
      compileWithTransformer((_, builder) => builder
        .setAttr("value", tensor)
        .setAttr("dtype", TensorType[A].tag)
      )

    def compileWithAllInputs: Builder[A, State with WithCompiler] =
      compileWithTransformer((ctx, builder) =>
        ctx.inputs.foldLeft(builder)((acc, next) => acc.addInput(next.output(0))))

    def compileWithInputList: Builder[A, State with WithCompiler] =
      compileWithTransformer((ctx, builder) => builder.addInputList(ctx.inputs.map(_.output(0)).toArray))

    def compileWithAttr(name: String, tp: TensorType[_]): Builder[A, State with WithCompiler] =
      compileWithTransformer((_, builder) => builder.setAttr(name, tp.tag))

    def compileWithAttr(name: String, str: String): Builder[A, State with WithCompiler] =
      compileWithTransformer((_, builder) => Option(str).fold(builder)(builder.setAttr(name, _)))

    def compileWithAttr(name: String, l: Long): Builder[A, State with WithCompiler] =
      compileWithTransformer((_, builder) => builder.setAttr(name, l))

    def compileWithControlInputs: Builder[A, State with WithCompiler] =
      compileWithTransformer((ctx, builder) => ctx.controls.foldLeft(builder)(_.addControlInput(_)))

    def build(implicit ev: State =:= Complete): Output[A] = {
      core.Output[A](name, Option(label).getOrElse(name), shape, inputs, controls, (context: OpContext[A]) => {
        val init = context.global.scope.env.opBuilder(context.op.name, context.label.toString)
        val transformed = transformers.foldLeft(init)((acc, next) => next(context, acc))
        transformed.build()
      })
    }
  }

  def name[A: TensorType](name: String): Builder[A, WithName] = {
    Builder[A, WithName](name, label = null, shape = null, inputs = Nil, controls = Nil, transformers = Nil)
  }
}