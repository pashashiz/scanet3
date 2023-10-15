package scanet.math.grad

import scanet.core.{Expr, Floating, Node, Numeric, Params, Shape}
import scanet.math.alg.kernels.syntax._

import scala.collection.immutable.Seq
import scala.collection.mutable

class GradCalc[A: Numeric, R: Floating](out: Expr[A]) {
  require(
    out.shape.isScalar,
    "gradient is supported on scalars only, " +
    "reduce the output with sum() or other operation")
  private val graph = out.asGraph
  private val cache = mutable.HashMap.empty[String, Expr[R]]
  def calc[B: Numeric](withRespectTo: Expr[B]): Expr[R] = {
    val leaf = graph.find(withRespectTo.ref)
    require(
      leaf.isDefined,
      s"cannot find a gradient with respect to $withRespectTo " +
      s"cause that input is not a part of the computation graph")
    def gradRec(node: Node[Expr[_]]): Expr[R] =
      cache.get(node.id) match {
        case Some(result) => result
        case None =>
          val result =
            if (node.isRoot) {
              ones[R](Shape())
            } else {
              val grads = node.outputs.map { edge =>
                val parent = edge.to
                val parentGrad = gradRec(parent)
                parent.value.localGrad(edge.index, parentGrad)
              }
              plus(grads.toList)
            }
          cache.put(node.id, result)
          result
      }
    leaf.map(gradRec).get
  }
}

class GradCalcOps[A: Numeric, B: Numeric](
    out: Expr[A],
    withRespectTo: Expr[B]) {
  def returns[R: Floating]: Expr[R] =
    new GradCalc[A, R](out).calc[B](withRespectTo)
}

class GradCalcSeqOps[A: Numeric, B: Numeric](
    out: Expr[A],
    withRespectTo: Seq[Expr[B]]) {
  def returns[R: Floating]: Seq[Expr[R]] = {
    val calc = new GradCalc[A, R](out)
    withRespectTo.map(calc.calc[B])
  }
}

class GradCalcParamsOps[A: Numeric, B: Numeric](
    out: Expr[A],
    withRespectTo: Params[Expr[B]]) {
  def returns[R: Floating]: Params[Expr[R]] = {
    val calc = new GradCalc[A, R](out)
    withRespectTo.mapValues(calc.calc[B])
  }
}

class GradOps[A: Numeric](expr: Expr[A]) {

  def grad[B: Numeric](withRespectTo: Expr[B]): GradCalcOps[A, B] =
    new GradCalcOps[A, B](expr, withRespectTo)

  def grad[B: Numeric](withRespectTo: Seq[Expr[B]]): GradCalcSeqOps[A, B] =
    new GradCalcSeqOps[A, B](expr, withRespectTo)

  def grad[B: Numeric](withRespectTo: Params[Expr[B]]): GradCalcParamsOps[A, B] =
    new GradCalcParamsOps[A, B](expr, withRespectTo)
}

trait GradSyntax {
  implicit def toGradOps[A: Numeric](expr: Expr[A]): GradOps[A] =
    new GradOps[A](expr)
}

object syntax extends GradSyntax
