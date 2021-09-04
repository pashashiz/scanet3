package org.scanet.core

import org.scanet.core.DefaultCompiler.{Ctx, Stage}
import org.scanet.math.{Floating, Numeric}
import org.scanet.native.RawTensors
import org.tensorflow
import org.tensorflow.types.family.TType

import scala.collection.immutable.Seq
import org.tensorflow.{Operation, OperationBuilder}

trait Expr[A] {
  def name: String
  def label: String = name
  def as(label: String): Expr[A] = Labeled(this, label)
  // revisit: maybe we can generate a unique identifier of a tensor
  def id: Option[String] = None
  def tpe: Option[TensorType[A]]
  def shape: Shape
  final def rank: Int = shape.rank
  final def isScalar: Boolean = shape.isScalar
  final def broadcastableBy(smaller: Expr[A]): Boolean = shape.broadcastableBy(smaller.shape)
  final def broadcastableAny(other: Expr[A]): Boolean = shape.broadcastableAny(other.shape)
  def inputs: Seq[Expr[_]]
  def controls: Seq[Expr[_]] = Seq.empty
  final def allInputs: Seq[Expr[_]] = inputs ++ controls

  def compiler: Compiler[A]
  // revisit: check if someone really need LabeledOperationOut instead of OperationOut
  final def compile(session: SessionState): (SessionState, LabeledOperationOut) =
    compiler(session, this)

  def localGrad: Grad[A] = error(s"gradient is not implemented for '$name' expr")
  final def localGrad[R: Floating: Numeric: TensorType](
      index: Int,
      parentGrad: Expr[R]): Expr[R] = {
    val grads = localGrad.calc[R](this, parentGrad)
    grads(index)
  }
  final def asGraph: DirectedGraph[Expr[_]] = {
    def fill(graph: DirectedGraph[Expr[_]], current: Expr[_]): DirectedGraph[Expr[_]] = {
      if (!graph.contains(current.toString)) {
        val withCurrent = graph :+ Node(current.toString, current)
        val withAll = current.inputs.foldLeft(withCurrent)((g, next) => fill(g, next))
        withAll.linkAll(current.inputs.map(node => (node.toString, current.toString)))
      } else {
        graph
      }
    }
    fill(DirectedGraph[Expr[_]](), this)
  }

  final override def toString: String = {
    val fullName = if (label == name) s"$name" else s"$label:$name"
    val child = id match {
      case Some(value) => s"($value)"
      case None        => if (inputs.nonEmpty) inputs.mkString("(", ", ", ")") else ""
    }
    val deps = if (controls.nonEmpty) s".depends(${controls.mkString(", ")})" else ""
    val tpeOrEmpty = tpe.map(t => s"[${t.show}]").getOrElse("")
    s"$fullName$child$deps$tpeOrEmpty:$shape"
  }
  final def address: String = super.hashCode().toString
  final override def hashCode(): Int = toString.hashCode
  final override def equals(obj: Any): Boolean = obj match {
    case other: Expr[_] => toString == other.toString
    case _              => false
  }
}

trait Compiler[A] extends ((SessionState, Expr[A]) => (SessionState, LabeledOperationOut))

case class DefaultCompiler[A](index: Option[Int], stages: Seq[Stage]) extends Compiler[A] {

  override def apply(session: SessionState, expr: Expr[A]): (SessionState, LabeledOperationOut) = {
    val cached = session.cache.get(expr.toString)
    cached.map(compiled => (session, compiled)).getOrElse(compile(session, expr))
  }

  private def compile(session: SessionState, expr: Expr[A]): (SessionState, LabeledOperationOut) = {
    val (sessionAfterInput, inputs) = compileAll(session, expr.inputs)
    val (sessionAfterControls, controls) = compileAll(sessionAfterInput, expr.controls)
    // operations in native graph have to have unique names,
    // so we just add an incremental index if there are any duplicates
    val uniqueLabel = Label(expr.label, sessionAfterControls.maxLabelIndex(expr.label) + 1)
    val builder = sessionAfterControls.scope.env.opBuilder(expr.name, uniqueLabel.toString)
    val ctx = Ctx(inputs, controls)
    val transformed =
      stages.foldLeft(builder)((builder, stage) => stage(ctx, builder))
    val compiled = LabeledOperationOut(OperationOut(transformed.build(), index), uniqueLabel)
    val newCache = sessionAfterControls.cache + (expr.toString -> compiled)
    val sessionAfterOutput = sessionAfterControls.copy(cache = newCache)
    (sessionAfterOutput, compiled)
  }

  private def compileAll(
      session: SessionState,
      inputs: Seq[Expr[_]]): (SessionState, Seq[OperationOut]) = {
    val (sessionAfterInput, outputs) =
      inputs.foldLeft((session, Seq[OperationOut]()))((acc, expr) => {
        val (currentSession, outs) = acc
        val (newSession, out) = expr.compile(currentSession)
        (newSession, out.operationOut +: outs)
      })
    (sessionAfterInput, outputs.reverse)
  }

  def withStage(stage: Stage): DefaultCompiler[A] = copy(stages = stages :+ stage)

  private def withInputs = withStage { (ctx: Ctx, builder: OperationBuilder) =>
    ctx.inputs.foldLeft(builder)((builder, next) => {
      builder.addInput(next.outputOrFail)
    })
  }

  private def withInputsAsList = withStage { (ctx: Ctx, builder: OperationBuilder) =>
    builder.addInputList(ctx.inputs.map(_.outputOrFail).toArray)
  }

  private def withControls = withStage { (ctx: Ctx, builder: OperationBuilder) =>
    ctx.controls.foldLeft(builder)((builder, next) => {
      builder.addControlInput(next.operation)
    })
  }

  def withValue[B: TensorType](tensor: Tensor[B]): DefaultCompiler[A] = withStage {
    (_: Ctx, builder: OperationBuilder) =>
      builder.setAttr("value", tensor.compact).setAttr("dtype", TensorType[B].tag)
  }

  def withAttr(name: String, value: TensorType[_]): DefaultCompiler[A] = withStage {
    (_: Ctx, builder: OperationBuilder) =>
      builder.setAttr(name, value.tag)
  }

  def withAttr(name: String, value: String): DefaultCompiler[A] = withStage {
    (_: Ctx, builder: OperationBuilder) =>
      builder.setAttr(name, value)
  }

  def withAttrs(attrs: Map[String, String]): DefaultCompiler[A] = withStage {
    (_: Ctx, builder: OperationBuilder) =>
      attrs.foldLeft(builder)((builder, next) => {
        val (key, value) = next
        builder.setAttr(key, value)
      })
  }

  def withAttr(name: String, value: Long): DefaultCompiler[A] = withStage {
    (_: Ctx, builder: OperationBuilder) =>
      builder.setAttr(name, value)
  }

  def withAttr(name: String, value: Float): DefaultCompiler[A] = withStage {
    (_: Ctx, builder: OperationBuilder) =>
      builder.setAttr(name, value)
  }

  def withAttr(name: String, value: Boolean): DefaultCompiler[A] = withStage {
    (_: Ctx, builder: OperationBuilder) =>
      builder.setAttr(name, value)
  }

  def withAttr(name: String, value: Shape): DefaultCompiler[A] = withStage {
    (_: Ctx, builder: OperationBuilder) =>
      builder.setAttr(name, RawTensors.toNativeShape(value))
  }
}

object DefaultCompiler {

  def apply[A](
      index: Option[Int] = Some(0),
      withInputs: Boolean = true,
      inputsAsList: Boolean = false,
      stages: Seq[Stage] = Seq.empty): DefaultCompiler[A] = {
    val compiler = new DefaultCompiler[A](index, stages)
    if (withInputs) {
      val withControls = compiler.withControls
      if (inputsAsList) withControls.withInputsAsList else withControls.withInputs
    } else {
      compiler
    }
  }

  case class Ctx(inputs: Seq[OperationOut], controls: Seq[OperationOut])
  type Stage = (Ctx, OperationBuilder) => OperationBuilder
}

trait Grad[A] {
  def calc[R: Numeric: Floating: TensorType](current: Expr[A], parentGrad: Expr[R]): Seq[Expr[R]]
}

case class Label(name: String, index: Int = 0) {
  require(name.nonEmpty, "name cannot be empty")
  require(index >= -1, "index should be positive or -1")
  override def toString: String = s"${name}_$index"
}

case class OperationOut(operation: Operation, index: Option[Int]) {
  def output: Option[tensorflow.Output[_ <: TType]] =
    index.map(i => operation.output(i).asInstanceOf[tensorflow.Output[_ <: TType]])
  def outputOrFail: tensorflow.Output[_ <: TType] =
    output.getOrElse(error(s"Operation output $this has unresolved output index"))
}
case class LabeledOperationOut(operationOut: OperationOut, label: Label)

case class Labeled[A](expr: Expr[A], override val label: String) extends Expr[A] {
  override def name: String = expr.name
  override def id: Option[String] = expr.id
  override def tpe: Option[TensorType[A]] = expr.tpe
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = expr.inputs
  override def controls: Seq[Expr[_]] = expr.controls
  override def compiler: Compiler[A] = expr.compiler
  override def localGrad: Grad[A] = expr.localGrad
}

case class Const[A: TensorType](tensor: Tensor[A]) extends Expr[A] {
  override val tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def name: String = "Const"
  override def shape: Shape = tensor.shape
  override def id: Option[String] = {
    val value =
      if (tensor.isScalar)
        tensor.toScalar.toString
      else if (tensor.rank == 1 && tensor.power <= 10)
        tensor.toArray.mkString(", ")
      else
        s"#${tensor.address}"
    Some(value)
  }
  override def inputs: Seq[Expr[_]] = Seq.empty
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = List()
  }
  override def compiler: Compiler[A] = DefaultCompiler().withValue(tensor)
}

object Const {

  class Ops[A: TensorType](val tensor: Tensor[A]) {
    def const: Expr[A] = Const(tensor)
  }

  def apply[A: TensorType](tensor: Tensor[A]): Expr[A] = new Const[A](tensor.compact)

  trait AllSyntax {
    implicit def scalarIsConstOps[A: TensorType](value: A): Ops[A] =
      new Ops(Tensor.scalar(value))
    implicit def tensorIsConstOps[A: TensorType](tensor: Tensor[A]): Ops[A] =
      new Ops(tensor)
  }

  object syntax extends AllSyntax
}
