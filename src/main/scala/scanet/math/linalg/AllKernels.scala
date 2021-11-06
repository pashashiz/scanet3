package scanet.math.linalg

import org.tensorflow.proto.framework.DataType
import scanet.core
import scanet.core.syntax._
import scanet.core.{Cast, DefaultCompiler, Expr, Grad, Shape, TakeOut, TakeOutUntyped, TensorType}
import scanet.math.{Floating, Numeric}

import scala.collection.immutable.Seq

case class Det[A: TensorType: Floating] private (tensor: Expr[A]) extends Expr[A] {
  require(
    tensor.rank >= 2,
    s"at least tensor with rank 2 is required but was passed a tensor with rank ${tensor.rank}")
  private val matrixShape = Shape(tensor.shape.dims.takeRight(2))
  require(
    matrixShape.dims.distinct.size <= 1,
    s"the last 2 dimensions should form a squared matrix, but was a matrix with shape $matrixShape")
  override def name: String = "MatrixDeterminant"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override val shape: Shape = Shape(tensor.shape.dims.dropRight(2))
  override def inputs: Seq[Expr[_]] = Seq(tensor)
  override def compiler: core.Compiler[A] = DefaultCompiler[A]()
}

/** NOTE: the result is a complex number which is not supported by JVM lib as of now
  * so we return Any Expr which should be casted after
  */
case class Eigen[A: TensorType: Floating] private (tensor: Expr[A], vectors: Boolean)
    extends Expr[(Any, Any)] {
  override def name: String = "Eig"
  override def tpe: Option[TensorType[(Any, Any)]] = None
  override def shape: Shape = Shape()
  override def inputs: Seq[Expr[_]] = Seq(tensor)
  override def compiler: core.Compiler[(Any, Any)] = DefaultCompiler[(Any, Any)](index = None)
    .withAttr("Tout", DataType.DT_COMPLEX64)
    .withAttr("compute_v", value = vectors)
}

trait AllKernels {

  /** Computes the determinant of one or more square matrices
    * @param tensor A The input is a tensor of shape [..., M, M] whose inner-most 2 dimensions form square matrices.
    *               The output is a tensor containing the determinants for all input submatrices [..., :, :]
    */
  def det[A: TensorType: Floating](tensor: Expr[A]): Expr[A] = Det(tensor)

  def eigenValues[A: TensorType: Floating](tensor: Expr[A]): Expr[A] =
    TakeOutUntyped(Eigen(tensor, vectors = false), 0, Shape(tensor.shape.dims.dropRight(1)))
      .castUnsafe[A]

  def eigenVectors[A: TensorType: Floating](tensor: Expr[A]): Expr[A] =
    TakeOutUntyped(Eigen(tensor, vectors = true), 1, tensor.shape).castUnsafe[A]

  def eigen[A: TensorType: Floating](tensor: Expr[A]): (Expr[A], Expr[A]) = {
    val e: Eigen[A] = Eigen(tensor, vectors = true)
    val values = TakeOutUntyped[Any](e, 0, Shape(tensor.shape.dims.dropRight(1))).castUnsafe[A]
    val vectors = TakeOutUntyped[Any](e, 1, tensor.shape).castUnsafe[A]
    (values, vectors)
  }
}

object kernels extends AllKernels {

  class LinalgFloatingOps[A: TensorType: Floating](expr: Expr[A]) {
    import scanet.math.linalg.{kernels => f}

    /** @see [[f.det]]
      */
    def det: Expr[A] = f.det(expr)

    def eigenValues: Expr[A] = f.eigenValues(expr)
    def eigenVectors: Expr[A] = f.eigenVectors(expr)
    def eigen: (Expr[A], Expr[A]) = f.eigen(expr)
  }

  trait AllSyntax extends AllKernels {
    implicit def toLinalgFloatingOps[A: TensorType: Floating](expr: Expr[A]): LinalgFloatingOps[A] =
      new LinalgFloatingOps[A](expr)
  }

  object syntax extends AllSyntax
}
