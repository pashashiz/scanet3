package org.scanet.research

import java.util.concurrent.atomic.AtomicInteger

import org.scanet.core.{DirectedGraph, Node}

object ReverseAD {

  val counter = new AtomicInteger(0)

  sealed trait Expr {

    val id: String
    def nextId: String = counter.getAndIncrement().toString
    def inputs: Seq[Expr]
    def eval: Float
    def localGrads: Map[Expr, Float]

    def directedGraph: DirectedGraph[Expr] = {
      def fill(graph: DirectedGraph[Expr], current: Expr): DirectedGraph[Expr] = {
        if (!graph.contains(current.id)) {
          val withCurrent = graph :+ Node(current.id, current)
          val withAll = current.inputs.foldLeft(withCurrent)((g, next) => fill(g, next))
          withAll.linkAll(current.inputs.map(node => (node.id, current.id)))
        } else {
          graph
        }
      }
      fill(DirectedGraph[Expr](), this)
    }

    def grad(input: Expr): Option[Float] = {
      val graph = directedGraph
      def gradRec(node: Node[Expr]): Float = {
        if (node.isRoot) {
          1.0f
        } else {
          node.outputs.map(parent => {
            val localGrad = parent.to.value.localGrads(node.value)
            val parentGrad = gradRec(parent.to)
            localGrad * parentGrad
          }).sum
        }
      }
      graph.find(input.id).map(gradRec)
    }
  }

  case class Const(value: Float) extends Expr {
    override val id: String = "const-" + nextId
    override def inputs: Seq[Expr] = Seq()
    override def eval: Float = value
    override def localGrads: Map[Expr, Float] = Map()
  }

  case class Plus(left: Expr, right: Expr) extends Expr {
    override val id: String = "plus-" + nextId
    override def inputs: Seq[Expr] = Seq(left, right)
    override def eval: Float = left.eval + right.eval
    override def localGrads: Map[Expr, Float] = {
      Map(left -> 1, right -> 1)
    }
  }

  case class Multiply(left: Expr, right: Expr) extends Expr {
    override val id: String = "multiply-" + nextId
    override def inputs: Seq[Expr] = Seq(left, right)
    override def eval: Float = left.eval * right.eval
    override def localGrads: Map[Expr, Float] = {
      Map(left -> right.eval, right -> left.eval)
    }
  }

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
    print(f.directedGraph)
    println(s"sample 1: ${f.eval}, ${f.grad(x)}") // 11.0, 2.0
  }

  def sample2(): Unit = {
    // f(x) = x * x + 2 * x + 5 -> df/dx = 2x + 2 | x = 3
    val a = Const(2)
    val b = Const(5)
    val x = Const(3)
    val f = Plus(Plus(Multiply(x, x), Multiply(a, x)), b)
    println(f.directedGraph)
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



