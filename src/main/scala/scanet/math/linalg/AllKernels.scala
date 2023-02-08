package scanet.math.linalg

import org.tensorflow.proto.framework.DataType
import scanet.core
import scanet.core.syntax._
import scanet.core.{DefaultCompiler, Expr, Floating, Shape, TakeOut, TakeOutUntyped, TensorType}

import scala.collection.immutable.Seq

case class Det[A: Floating](tensor: Expr[A]) extends Expr[A] {
  tensor.requireSquareMatrixTail
  override def name: String = "MatrixDeterminant"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override val shape: Shape = Shape(tensor.shape.dims.dropRight(2))
  override def inputs: Seq[Expr[_]] = Seq(tensor)
  override def compiler: core.Compiler[A] = DefaultCompiler[A]()
}

/** Notes:
  *  - There are 2 outputs
  *  - The result is a complex number which is not supported by JVM lib as of now
  *    so we return Any Expr which should be casted after
  */
case class Eigen[A: Floating](tensor: Expr[A], vectors: Boolean) extends Expr[(Any, Any)] {
  tensor.requireSquareMatrixTail
  override def name: String = "Eig"
  override def tpe: Option[TensorType[(Any, Any)]] = None
  override def shape: Shape = Shape()
  override def inputs: Seq[Expr[_]] = Seq(tensor)
  override def compiler: core.Compiler[(Any, Any)] = DefaultCompiler[(Any, Any)](index = None)
    .withAttr("Tout", DataType.DT_COMPLEX64)
    .withAttr("compute_v", value = vectors)
}

/** Notes:
  * - There are 2 outputs
  */
case class Qr[A: Floating](tensor: Expr[A], fullMatrices: Boolean) extends Expr[A] {
  tensor.requireAtLestRank(2)
  override def name: String = "Qr"
  override def tpe: Option[TensorType[A]] = None
  override def shape: Shape = Shape()
  override def inputs: Seq[Expr[_]] = Seq(tensor)
  override def compiler: core.Compiler[A] = DefaultCompiler[A](index = None)
    .withAttr("full_matrices", value = fullMatrices)
}

case class DiagPart[A: TensorType](tensor: Expr[A]) extends Expr[A] {
  private val (left, right) = tensor.shape.splitAtHalf
  require(
    left == right,
    s"tensor should have symmetric shape as [D1,..., Dk, D1,..., Dk], but was ${tensor.shape}")
  override def name: String = "DiagPart"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override val shape: Shape = left
  override def inputs: Seq[Expr[_]] = Seq(tensor)
  override def compiler: core.Compiler[A] = DefaultCompiler[A]()
}

trait AllKernels {

  /** Computes the determinant of one or more square matrices
    * @param tensor A The input is a tensor of shape [..., M, M] whose inner-most 2 dimensions form square matrices.
    *               The output is a tensor containing the determinants for all input submatrices [..., :, :]
    */
  def det[A: Floating](tensor: Expr[A]): Expr[A] = Det(tensor)

  /** Computes the eigenvalues for a matrix or a batch of matrices (D3)
    *
    * @param tensor Tensor of shape [..., N, N].
    *               Only the lower triangular part of each inner inner matrix is referenced.
    */
  def eigenValues[A: Floating](tensor: Expr[A]): Expr[A] =
    TakeOutUntyped(Eigen(tensor, vectors = false), 0, Shape(tensor.shape.dims.dropRight(1)))
      .castUnsafe[A]

  /** Computes the eigenvectors for a matrix or a batch of matrices (D3)
    *
    * @param tensor Tensor of shape [..., N, N].
    *               Only the lower triangular part of each inner inner matrix is referenced.
    */
  def eigenVectors[A: Floating](tensor: Expr[A]): Expr[A] =
    TakeOutUntyped(Eigen(tensor, vectors = true), 1, tensor.shape).castUnsafe[A]

  /** @see [[eigenValues]] and [[eigenVectors]] */
  def eigen[A: Floating](tensor: Expr[A]): (Expr[A], Expr[A]) = {
    val e: Eigen[A] = Eigen(tensor, vectors = true)
    val values = TakeOutUntyped[Any](e, 0, Shape(tensor.shape.dims.dropRight(1))).castUnsafe[A]
    val vectors = TakeOutUntyped[Any](e, 1, tensor.shape).castUnsafe[A]
    (values, vectors)
  }

  /** Computes the QR decomposition of each inner matrix in tensor such that
    * `tensor [..., :, :] = q[..., :, :] * r[..., :,:]`
    *
    * @param tensor A tensor of shape `[..., M, N]` whose inner-most 2 dimensions form matrices of size `[M, N]`.
    *               Let `P` be the minimum of `M` and `N`.
    * @param fullMatrices If `true`, compute full-sized q and r, if `false`, compute only the leading `P` columns of q.
    * @return A tuple of tensors (q, r). The result shapes:
    *         - if `fullMatrices` is `false` then `q = [M, P]` and `r = [P, N]`,
    *         - if `fullMatrices` is `true` then `q = [M, M]` and `r = [M, N]`
    */
  def qr[A: Floating](tensor: Expr[A], fullMatrices: Boolean = false): (Expr[A], Expr[A]) = {
    val qr: Qr[A] = Qr(tensor, fullMatrices = fullMatrices)
    val base = tensor.shape.dropRight(2)
    val List(m, n) = tensor.shape.takeRight(2).dims
    val p = math.min(m, n)
    val (qShape, rShape) =
      if (!fullMatrices) (base ++ Shape(m, p), base ++ Shape(p, n))
      else (base ++ Shape(m, m), base ++ Shape(m, n))
    val q = TakeOut[A](qr, 0, qShape)
    val r = TakeOut[A](qr, 1, rShape)
    (q, r)
  }

  /** This operation returns a tensor with the diagonal part of the input.
    *
    * The diagonal part is computed as follows:
    * Assume input has dimensions `[D1,..., Dk, D1,..., Dk]`,
    * then the output is a tensor of rank k with dimensions `[D1,..., Dk]` where:
    * `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.
    *
    * @param tensor Tensor of shape `[D1,..., Dk, D1,..., Dk]`
    * @return Diagonal tensor of shape `[D1,..., Dk]`
    */
  def diagPart[A: TensorType](tensor: Expr[A]): Expr[A] = DiagPart(tensor)
}

object kernels extends AllKernels {

  class LinalgFloatingOps[A: Floating](expr: Expr[A]) {
    import scanet.math.linalg.{kernels => f}

    /** @see [[f.det]] */
    def det: Expr[A] = f.det(expr)

    /** @see [[f.eigenValues]] */
    def eigenValues: Expr[A] = f.eigenValues(expr)

    /** @see [[f.eigenVectors]] */
    def eigenVectors: Expr[A] = f.eigenVectors(expr)

    /** @see [[f.eigen]] */
    def eigen: (Expr[A], Expr[A]) = f.eigen(expr)

    /** @see [[f.qr]] */
    def qr(fullMatrices: Boolean = false): (Expr[A], Expr[A]) = f.qr(expr, fullMatrices)
  }

  class LinalgTensorOps[A: TensorType](expr: Expr[A]) {
    import scanet.math.linalg.{kernels => f}

    /** @see [[f.diagPart]] */
    def diagPart: Expr[A] = f.diagPart(expr)
  }

  trait AllSyntax extends AllKernels {
    implicit def toLinalgFloatingOps[A: Floating](expr: Expr[A]): LinalgFloatingOps[A] =
      new LinalgFloatingOps[A](expr)

    implicit def toLinalgTensorOps[A: TensorType](expr: Expr[A]): LinalgTensorOps[A] =
      new LinalgTensorOps[A](expr)
  }

  object syntax extends AllSyntax
}
