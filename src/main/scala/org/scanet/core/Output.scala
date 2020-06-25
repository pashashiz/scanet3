package org.scanet.core

import java.util.UUID

import org.scanet.core
import org.scanet.core.Output.BuilderState._
import org.scanet.core.Output._
import org.scanet.native.NativeTensorOps._
import org.tensorflow.{Operation, OperationBuilder}

case class Output[A: TensorType](
      id: String = UUID.randomUUID().toString,
      name: String,
      label: String,
      shape: Shape,
      inputs: List[Output[_]],
      controls: List[Output[_]],
      compiler: CompileContext[A] => Operation,
      localGradF: GradContext[A] => List[Output[Float]]) {

  def withId(newId: String): Output[A] = copy(id = newId)

  def withId(newId: Option[String]): Output[A] = newId.map(withId).getOrElse(this)

  def rank: Int = shape.rank

  def isScalar: Boolean = shape.isScalar

  def broadcastableBy(smaller: Output[A]): Boolean = shape.broadcastableBy(smaller.shape)

  def broadcastableAny(other: Output[A]): Boolean = shape.broadcastableAny(other.shape)

  def compile(context: SessionState): (SessionState, Compiled) = {
    val (contextAfterInput, outputs) = compileInputs(inputs, context)
    val (contextAfterControls, controlOuts) = compileInputs(controls, contextAfterInput)
    val uniqueLabel = Label(label, contextAfterControls.maxLabelIndex(label) + 1)
    val output = compiler(CompileContext(contextAfterControls, this, uniqueLabel, outputs.reverse, controlOuts))
    val compiled = (uniqueLabel, output)
    val newCache = contextAfterControls.cache + (id -> compiled)
    val contextAfterOutput = contextAfterControls.copy(cache = newCache)
    (contextAfterOutput, compiled)
  }

  private def compileInputs(in: List[Output[_]], context: SessionState): (SessionState, List[Operation]) = {
    val (contextAfterInput, outputs) = in.foldLeft((context, List[Operation]()))(
      (acc, op) => {
        val (currentContext, outs) = acc
        val (newContext, out) = op.findOrCompile(currentContext)
        (newContext, out._2 :: outs)
      })
    (contextAfterInput, outputs)
  }

  def findOrCompile(context: SessionState): (SessionState, Compiled) = {
    context.cache.get(id)
      .map(compiled => (context, compiled))
      .getOrElse {compile(context)}
  }

  def upstreamOptions: List[Output[_]] = {
    inputs.flatMap(op => op.upstreamOptions)
  }

  def localGrad(index: Int, parentGrad: Output[Float]): Output[Float] = {
    localGradF(GradContext(this, parentGrad))(index)
  }

  def asGraph: DirectedGraph[Output[_]] = {
    def fill(graph: DirectedGraph[Output[_]], current: Output[_]): DirectedGraph[Output[_]] = {
      val withCurrent = graph :+ Node(current.id, current)
      val withAll = current.inputs.foldLeft(withCurrent)((g, next) => fill(g, next))
      withAll.linkAll(current.inputs.map(node => (node.id, current.id)))
    }
    fill(DirectedGraph[Output[_]](), this)
  }

  override def toString: String = {
    val args = if (inputs.nonEmpty) s"(${inputs.mkString(", ")})" else ""
    val fullName = if (label == name) s"$name" else s"$label:$name"
    s"$fullName$args[${TensorType[A].show}]:$shape"
  }

  override def hashCode(): Int = id.hashCode

  override def equals(obj: Any): Boolean = obj match {
    case other: Output[_] => id == other.id
    case _ => false
  }
}

object Output {

  case class Label(value: String, index: Int = 0) {
    require(value.nonEmpty, "name cannot be empty")
    require(index >= -1, "index should be positive or -1")
    override def toString: String = s"${value}_$index"
  }

  type Compiled = (Label, Operation)

  case class GradContext[A](current: Output[A], parentGrad: Output[Float])

  case class CompileContext[A: TensorType](
        state: SessionState,
        op: Output[A],
        label: Label,
        inputs: List[Operation],
        controls: List[Operation])

  sealed trait BuilderState
  object BuilderState {
    sealed trait WithName extends BuilderState
    sealed trait WithShape extends BuilderState
    sealed trait WithCompiler extends BuilderState
    type Complete = WithName with WithShape with WithCompiler
    type Transformer[A] = (CompileContext[A], OperationBuilder) => OperationBuilder
  }

  case class Builder[A: TensorType, State <: BuilderState](
        name: String,
        label: String = null,
        shape: Shape = null,
        inputs: List[Output[A]] = Nil,
        controls: List[Output[A]] = Nil,
        transformers: List[Transformer[A]] = Nil,
        localGradF: GradContext[A] => List[Output[Float]] = null) {

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
      compileWithAllInputsAtIndex(0)

    def compileWithAllInputsAtIndex(idx: Int): Builder[A, State with WithCompiler] =
      compileWithTransformer((ctx, builder) =>
        ctx.inputs.foldLeft(builder)((acc, next) => acc.addInput(next.output(idx))))

    def compileWithInputList: Builder[A, State with WithCompiler] =
      compileWithTransformer((ctx, builder) => builder.addInputList(ctx.inputs.map(_.output(0)).toArray))

    def compileWithAttr(name: String, tp: TensorType[_]): Builder[A, State with WithCompiler] =
      compileWithTransformer((_, builder) => builder.setAttr(name, tp.tag))

    def compileWithAttr(name: String, str: String): Builder[A, State with WithCompiler] =
      compileWithTransformer((_, builder) => Option(str).fold(builder)(builder.setAttr(name, _)))

    def compileWithAttrs(attrs: Map[String, String]): Builder[A, State with WithCompiler] =
      compileWithTransformer((_, builder) => attrs.foldLeft(builder) { case (b, (k, v)) => b.setAttr(k, v) })

    def compileWithAttr(name: String, l: Long): Builder[A, State with WithCompiler] =
      compileWithTransformer((_, builder) => builder.setAttr(name, l))

    def compileWithAttr(name: String, shape: Shape): Builder[A, State with WithCompiler] =
      compileWithTransformer((_, builder) => builder.setAttr(name, shape))

    def compileWithControlInputs: Builder[A, State with WithCompiler] =
      compileWithTransformer((ctx, builder) => ctx.controls.foldLeft(builder)(_.addControlInput(_)))

    def localGrad(f: GradContext[A] => List[Output[Float]]): Builder[A, State] =
      copy(localGradF = f)

    def build(implicit ev: State =:= Complete): Output[A] = {
      core.Output[A](
        name = name,
        label = Option(label).getOrElse(name),
        shape = shape,
        inputs = inputs,
        controls = controls,
        compiler = (context: CompileContext[A]) => {
          val init = context.state.scope.env.opBuilder(context.op.name, context.label.toString)
          val transformed = transformers.foldLeft(init)((acc, next) => next(context, acc))
          transformed.build()
        },
        localGradF = Option(localGradF).getOrElse((_: GradContext[A]) => error(s"gradient is not implemented for '$name' operator")))
    }
  }

  def name[A: TensorType](name: String): Builder[A, WithName] = {
    Builder[A, WithName](name)
  }
}