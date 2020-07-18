package org.scanet.core

import org.scanet.core
import org.scanet.core.Output.BuilderState._
import org.scanet.core.Output._
import org.scanet.math.{Floating, Numeric}
import org.scanet.native.NativeTensorOps._
import org.tensorflow.{Operation, OperationBuilder}

case class Output[A: TensorType](
      name: String,
      label: String,
      shape: Shape,
      id: Output[A] => Option[String],
      index: Int,
      inputs: List[Output[_]],
      controls: List[Output[_]],
      compiler: CompileContext[A] => Operation,
      gradF: Grad[A]) {

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
    val newCache = contextAfterControls.cache + (toString -> compiled)
    val contextAfterOutput = contextAfterControls.copy(cache = newCache)
    (contextAfterOutput, compiled)
  }

  private def compileInputs(in: List[Output[_]], context: SessionState): (SessionState, List[(Operation, Int)]) = {
    val (contextAfterInput, outputs) = in.foldLeft((context, List[(Operation, Int)]()))(
      (acc, op) => {
        val (currentContext, outs) = acc
        val (newContext, out) = op.findOrCompile(currentContext)
        (newContext, (out._2, op.index) :: outs)
      })
    (contextAfterInput, outputs)
  }

  def findOrCompile(context: SessionState): (SessionState, Compiled) = {
    context.cache.get(toString)
      .map(compiled => (context, compiled))
      .getOrElse {compile(context)}
  }

  def upstreamOptions: List[Output[_]] = {
    inputs.flatMap(op => op.upstreamOptions)
  }

  def localGrad[R: Floating: Numeric: TensorType](index: Int, parentGrad: Output[R]): Output[R] = {
    val grads = gradF.calc[R](this, parentGrad)
    grads(index)
  }

  def asGraph: DirectedGraph[Output[_]] = {
    def fill(graph: DirectedGraph[Output[_]], current: Output[_]): DirectedGraph[Output[_]] = {
      if (!graph.contains(current.toString)) {
        val withCurrent = graph :+ Node(current.toString, current)
        val withAll = current.inputs.foldLeft(withCurrent)((g, next) => fill(g, next))
        withAll.linkAll(current.inputs.map(node => (node.toString, current.toString)))
      } else {
        graph
      }
    }
    fill(DirectedGraph[Output[_]](), this)
  }

  override def toString: String = {
    val fullName = if (label == name) s"$name" else s"$label:$name"
    val child = id(this) match {
      case Some(value) => s"($value)"
      case None =>
        if (inputs.nonEmpty) {
          val args = inputs.map(output => {
            // we need to know which output we are taking
            val suffix = if (output.index > 0) s"[$index]" else ""
            output + suffix
          }).mkString(", ")
          s"($args)"
        }
        else {
          ""
        }
    }
    val deps = if (controls.nonEmpty) s".depends(${controls.mkString(", ")})" else ""
    s"$fullName$child$deps[${TensorType[A].show}]:$shape"
  }

  def address: String = super.hashCode().toString

  override def hashCode(): Int = toString.hashCode

  override def equals(obj: Any): Boolean = obj match {
    case other: Output[_] => toString == other.toString
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

//  case class GradContext[A, R: Floating: Numeric: TensorType](current: Output[A], parentGrad: Output[R])

  trait Grad[A] {
    def calc[R: Numeric: Floating: TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]]
  }

  case class CompileContext[A: TensorType](
        state: SessionState,
        op: Output[A],
        label: Label,
        inputs: List[(Operation, Int)],
        controls: List[(Operation, Int)])

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
        id: Output[A] => Option[String] = (_: Output[A]) => None,
        index: Int = 0,
        inputs: List[Output[A]] = Nil,
        controls: List[Output[A]] = Nil,
        transformers: List[Transformer[A]] = Nil,
        grad: Grad[A] = null) {

    def label(label: String): Builder[A, State] = copy(label = label)

    def shape(shape: Shape): Builder[A, State with WithShape] = copy(shape = shape)

    def id(id: Output[A] => String): Builder[A, State] = copy(id = output => Some(id(output)))

    def inputs(inputs: Output[_]*): Builder[A, State] = {
      copy(inputs = inputs.toList.asInstanceOf[List[Output[A]]])
    }

    def index(i: Int): Builder[A, State] = copy(index = i)

    def controlInputs(controls: Output[_]*): Builder[A, State] = {
      copy(controls = controls.toList.asInstanceOf[List[Output[A]]])
    }

    def compileWithTransformer(f: Transformer[A]): Builder[A, State with WithCompiler] =
      copy(transformers = f :: transformers)

    def compileWithValue(tensor: Tensor[A]): Builder[A, State with WithCompiler] =
      compileWithTransformer((_, builder) => builder
        .setAttr("value", tensor.compact)
        .setAttr("dtype", TensorType[A].tag)
      )

    def compileWithAllInputs: Builder[A, State with WithCompiler] =
      compileWithTransformer((ctx, builder) => {
        ctx.inputs.foldLeft(builder)((acc, next) => {
          val (operation, index) = next
          acc.addInput(operation.output(index))
        })
      })

    def compileWithInputList: Builder[A, State with WithCompiler] =
      compileWithTransformer((ctx, builder) =>
        builder.addInputList(
          ctx.inputs.map(next => {
            val (operation, index) = next
            operation.output(index)
          }).toArray))

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

    def compileWithAttr(name: String, shape: Boolean): Builder[A, State with WithCompiler] =
      compileWithTransformer((_, builder) => builder.setAttr(name, shape))


    def compileWithControlInputs: Builder[A, State with WithCompiler] =
      compileWithTransformer((ctx, builder) => ctx.controls.foldLeft(builder)((acc, next) => {
        acc.addControlInput(next._1)
      }))

    def localGrad(grad: Grad[A]): Builder[A, State] =
      copy(grad = grad)

    def build(implicit ev: State =:= Complete): Output[A] = {
      core.Output[A](
        name = name,
        label = Option(label).getOrElse(name),
        shape = shape,
        id = id,
        index = index,
        inputs = inputs,
        controls = controls,
        compiler = (context: CompileContext[A]) => {
          val init = context.state.scope.env.opBuilder(context.op.name, context.label.toString)
          val transformed = transformers.foldLeft(init)((acc, next) => next(context, acc))
          transformed.build()
        },
        gradF = Option(grad).getOrElse(new Grad[A] {
          override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
            error(s"gradient is not implemented for '$name' operator")
          }
        }))
    }

    def build2(implicit ev: State =:= Complete): (Output[A], Output[A]) = {
      // NOTE: Output index IS NOT a part of a unique id of the output
      // so if the same output is referenced few times the same operation will be reused via session cache.
      // However, output index IS a part of the argument which means that if 2 similar
      // operations consume the same output but with different indexes
      // they will have different unique ids and cache will not work for them which we expect
      val output = build
      val first = Output.name[A]("Identity")
        .inputs(output)
        .shape(Shape())
        .compileWithAllInputs
        .build
      val second = Output.name[A]("Identity")
        .inputs(output.copy(index = 1))
        .shape(Shape())
        .compileWithAllInputs
        .build
      (first, second)
    }
  }

  def name[A: TensorType](name: String): Builder[A, WithName] = {
    Builder[A, WithName](name)
  }
}