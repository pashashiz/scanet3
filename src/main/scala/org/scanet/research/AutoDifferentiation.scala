package org.scanet.research

import java.util.UUID

object AutoDifferentiation {

  def main(args: Array[String]): Unit = {
    sample1()
    sample2()
  }

  def sample1(): Unit = {
    // c = 2x + 5 | x = 3
    val a = Const(2)
    val b = Const(5)
    val x = Const(3)
    val c = Plus(Multiply(a, x), b)
    println(c.eval) // 11
    // df/dx ?
    println(c.grad(x)) // 2
  }

  def sample2(): Unit = {
    // c = x * x + 2 * x + 5 | x = 3
    val a = Const(2)
    val b = Const(5)
    val x = Const(3)
    // c = 2x + 5
    val c = Plus(Plus(Multiply(x, x), Multiply(a, x)), b)
    println(c.eval) // 20
    // df/dx ?
    println(c.grad(x)) // 8
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
  // 2 -> 0
  override def grad(input: Expr): Float =
    if (this == input) 1.0f
    else 0.0f
}

case class Plus(left: Expr, right: Expr) extends Expr {
  override def id: String = nextId
  override def eval: Float = left.eval + right.eval
  // x + x -> 1 + 1
  // 2 + x -> 0 + 1
  // 2 + 2 -> 0 + 0
  override def grad(input: Expr): Float = {
    left.grad(input) + right.grad(input)
  }
}

case class Multiply(left: Expr, right: Expr) extends Expr {
  override def id: String = nextId
  override def eval: Float = left.eval * right.eval
  // 2 * 1 = 2 * 0 + 1 * 0
  // 2 * x = 2 * 1 + x * 0
  // x * x = x * 1 + x * 1
  override def grad(input: Expr): Float = {
    left.eval * right.grad(input) + right.eval * left.grad(input)
  }
}
