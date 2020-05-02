package org.scanet.research

import java.util.UUID

object AutoDifferentiation {

  def main(args: Array[String]): Unit = {
    sample1()
    sample2()
    sample3()
    sample4()
  }

  def sample1(): Unit = {
    // f(x) = 2x + 5 -> df/dx = 2 | x = 3
    val a = Const(2)
    val b = Const(5)
    val x = Const(3)
    val f = Plus(Multiply(a, x), b)
    println(s"sample 1: ${f.eval}, ${f.grad(x)}") // 11.0, 2.0
  }

  def sample2(): Unit = {
    // f(x) = x * x + 2 * x + 5 -> df/dx = 2x + 2 | x = 3
    val a = Const(2)
    val b = Const(5)
    val x = Const(3)
    val f = Plus(Plus(Multiply(x, x), Multiply(a, x)), b)
    println(s"sample 2: ${f.eval}, ${f.grad(x)}") // 20.0, 8.0

  }

  def sample3(): Unit = {
    // f(x) = x * (x + 5) -> df/dx = 2x + 5 | x = 3
    val a = Const(5)
    val x = Const(3)
    val f = Multiply(x, Plus(x, a))
    println(s"sample 3: ${f.eval}, ${f.grad(x)}") // 24, 11
  }

  def sample4(): Unit = {
    // f(x) = x * x * x -> df/dx = 3x^2 | x = 4
    val x = Const(4)
    val f = Multiply(Multiply(x, x), x)
    println(s"sample 4: ${f.eval}, ${f.grad(x)}") // 64, 48
  }
}

sealed trait Expr {
  def id: String
  def nextId: String = UUID.randomUUID().toString
  def eval: Float
  def grad(input: Expr): Float
}

case class Const(value: Float) extends Expr {
  override def id: String = nextId
  override def eval: Float = value
  // x -> 1
  // a -> 0
  override def grad(input: Expr): Float =
    if (this == input) 1.0f
    else 0.0f
}

case class Plus(left: Expr, right: Expr) extends Expr {
  override def id: String = nextId
  override def eval: Float = left.eval + right.eval
  // x + x -> 1 + 1
  // a + x -> 0 + 1
  // a + b -> 0 + 0
  override def grad(input: Expr): Float = {
    left.grad(input) + right.grad(input)
  }
}

case class Multiply(left: Expr, right: Expr) extends Expr {
  override def id: String = nextId
  override def eval: Float = left.eval * right.eval
  // a * b = a * 0 + b * 0
  // a * x = a * 1 + x * 0
  // x * x = x * 1 + x * 1
  override def grad(input: Expr): Float = {
    left.eval * right.grad(input) + right.eval * left.grad(input)
  }
}
