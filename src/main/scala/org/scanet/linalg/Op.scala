package org.scanet.linalg

import java.util.UUID

import org.scanet.core.Numeric
import org.tensorflow.Output
import org.tensorflow.op.{Scope => NativeScope}

case class Context(scope: NativeScope, cache: Map[String, Output[AnyRef]])

case class Op[A: Numeric](name: String, shape: Shape, inputs: List[Op[A]], compiler: (Context, List[Output[A]]) => Output[A]) {

  require(name.nonEmpty, "name cannot be empty")

  val id: String = UUID.randomUUID().toString

  def compile(context: Context): (Context, Output[A]) = {
    val (context1, outputs) = inputs.foldLeft((context, List[Output[A]]()))(
      (acc, op) => {
        val (currentContext, outs) = acc
        val (newContext, out) = op.findOrCompile(currentContext)
        (newContext, out::outs)
      })
    val output = compiler.apply(context1, outputs.reverse)
    val newCache = context1.cache + (id -> output.asInstanceOf[Output[AnyRef]])
    val context2 = context1.copy(cache = newCache)
    (context2, output)
  }

  def findOrCompile(context: Context): (Context, Output[A]) = {
    context.cache.get(id)
      .map(operand => (context, operand.asInstanceOf[Output[A]]))
      .getOrElse {compile(context)}
  }

  override def toString: String = {
    val args = if (inputs.nonEmpty) s"(${inputs.mkString(", ")})" else ""
    name + args
  }

  def eval: Tensor[A] = Session.run(this)
}

object Op {

  class Arg0Builder[A: Numeric](name: String, shape: Shape) {
    def compileWith(f: Context => Output[A]): Op[A]  = {
      Op(name, shape, Nil, (context, _) => f(context))
    }
  }

  class Arg1Builder[A: Numeric](name: String, shape: Shape, arg1: Op[A]) {
    def compileWith(f: (Context, Output[A]) => Output[A]): Op[A]  = {
      Op(name, shape, List(arg1), (context, inputs) => f(context, inputs.head))
    }
  }

  class Arg2Builder[A: Numeric](name: String, shape: Shape, arg1: Op[A], arg2: Op[A]) {
    def compileWith(f: (Context, Output[A], Output[A]) => Output[A]): Op[A]  = {
      Op(name, shape, List(arg1, arg2), (context, inputs) => f(context, inputs.head, inputs(1)))
    }
  }

  def build[A: Numeric](name: String, shape: Shape): Arg0Builder[A] = new Arg0Builder[A](name, shape)
  def build[A: Numeric](name: String, shape: Shape, arg1: Op[A]): Arg1Builder[A] = new Arg1Builder[A](name, shape, arg1)
  def build[A: Numeric](name: String, shape: Shape, arg1: Op[A], arg2: Op[A]): Arg2Builder[A] = new Arg2Builder[A](name, shape, arg1, arg2)

  // def of(name: String, shape: Shape)

  def const[A: Numeric](value: A): Op[A] =
    const(Tensor.scalar[A](value))

  def const[A: Numeric](value: A, name: String): Op[A] =
    const(Tensor.scalar[A](value), name)

  def const[A: Numeric](tensor: Tensor[A], name: String = "const"): Op[A] =
    Op.build(name, tensor.shape)
      .compileWith(context => {
        context.scope.env
          .opBuilder("Const", name)
          .setAttr("value", tensor)
          .setAttr("dtype", Numeric[A].tag)
          .build
          .output(0)
      })

  def plus[A: Numeric](left: Op[A], right: Op[A], name: String = "plus"): Op[A] = {
    require(left.shape.last == right.shape.last || left.shape.isScalar || right.shape.isScalar,
      s"tensors with shapes ${left.shape} and ${right.shape} cannot be added, " +
        "either last dimensions should equal or one of the tensors should be a scalar")
    Op.build(name, left.shape, left, right)
      .compileWith((context, leftOut, rightOut) => {
        context.scope.env.opBuilder("Add", name)
          .addInput(leftOut)
          .addInput(rightOut)
          .build()
          .output(0)
      })
  }
}

