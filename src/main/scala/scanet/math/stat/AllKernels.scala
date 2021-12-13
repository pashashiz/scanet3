package scanet.math.stat

import scanet.core.{Expr, Numeric, TensorType}
import scanet.math.alg.kernels.syntax._

import scala.collection.immutable.Seq

trait AllKernels {

  /** Computes the variance of elements across all dimensions of a tensor.
    * The result is a scalar tensor
    */
  def variance[A: Numeric](tensor: Expr[A]): Expr[A] =
    variance(tensor, 0 until tensor.rank)

  /** Computes the variance of elements across dimensions of a tensor
    *
    * Reduces input `tensor` along the dimensions given in `axis`.
    * Unless `keepDims` is `true`, the rank of the tensor is reduced by 1 for each
    * of the entries in `axis`, which must be unique. If `keepDims` is `true`, the
    * reduced dimensions are retained with length 1.
    */
  def variance[A: Numeric](
      tensor: Expr[A],
      axis: Seq[Int],
      keepDims: Boolean = false): Expr[A] = {
    val mean = tensor.mean(axis, keepDims = true)
    (tensor - mean).sqr.mean(axis, keepDims)
  }

  /** Computes the standard deviation of elements across all dimensions of a tensor.
    * The result is a scalar tensor
    */
  def std[A: Numeric](tensor: Expr[A]): Expr[A] =
    std(tensor, 0 until tensor.rank)

  /** Computes the standard deviation of elements across dimensions of a tensor
    *
    * Reduces input `tensor` along the dimensions given in `axis`.
    * Unless `keepDims` is `true`, the rank of the tensor is reduced by 1 for each
    * of the entries in `axis`, which must be unique. If `keepDims` is `true`, the
    * reduced dimensions are retained with length 1.
    */
  def std[A: Numeric](
      tensor: Expr[A],
      axis: Seq[Int],
      keepDims: Boolean = false): Expr[A] =
    variance(tensor, axis, keepDims).sqrt
}

object kernels extends AllKernels {

  class NumericOps[A: Numeric](expr: Expr[A]) {
    import scanet.math.stat.{kernels => f}

    /** @see [[f.variance]] */
    def variance: Expr[A] = f.variance(expr)

    /** @see [[f.variance]] */
    def variance(axis: Seq[Int], keepDims: Boolean = false): Expr[A] =
      f.variance(expr, axis, keepDims)

    /** @see [[f.std]] */
    def std: Expr[A] = f.std(expr)

    /** @see [[f.std]] */
    def std(axis: Seq[Int], keepDims: Boolean = false): Expr[A] =
      f.std(expr, axis, keepDims)
  }

  trait AllSyntax extends AllKernels {
    implicit def toStatKernelNumericOps[A: Numeric](expr: Expr[A]): NumericOps[A] =
      new NumericOps[A](expr)
  }

  object syntax extends AllSyntax
}
