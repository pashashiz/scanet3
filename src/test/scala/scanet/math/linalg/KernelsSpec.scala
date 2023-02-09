package scanet.math.linalg

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec
import scanet.core.{Shape, Tensor}
import scanet.math.syntax._

class KernelsSpec extends AnyWordSpec with Matchers {

  "determinant" should {

    "be calculated for square matrix" in {
      val matrix = Tensor.matrix(Array(3f, 4f), Array(-1f, 2f)).const
      matrix.det.eval shouldBe Tensor.scalar(10f)
    }

    "be calculated for multiple matrices" in {
      val matrix =
        Tensor
          .apply(
            Array(
              // first
              3f, 4f, -1f, 2f,
              // second
              1f, 4f, -3f, 6f),
            Shape(2, 2, 2))
          .const
      matrix.det.eval shouldBe Tensor.vector(10f, 18f)
    }

    "fail for vector or scalar" in {
      the[IllegalArgumentException] thrownBy {
        val vector = Tensor.vector(1f, 2f, 3f).const
        vector.det.eval
      } should have message "requirement failed: at least tensor with rank 2 is required but was passed a tensor with rank 1"
    }

    "fail for non squared matrix" in {
      the[IllegalArgumentException] thrownBy {
        val matrix = Tensor.matrix(Array(3f, 4f, 4f), Array(-1f, 2f, 2f)).const
        matrix.det.eval
      } should have message "requirement failed: the last 2 dimensions should form a squared matrix, but was a matrix with shape (2, 3)"
    }
  }

  "eigen" should {

    val matrix = Tensor.matrix(Array(3f, 0f), Array(1f, 2f)).const

    "compute eigen values" in {
      matrix.round.eigenValues.roundAt(2).eval shouldBe Tensor.vector(3f, 2f)
    }

    "compute eigen vectors" in {
      matrix.eigenVectors.roundAt(2).eval shouldBe
      Tensor.matrix(
        Array(0.71f, 0.0f),
        Array(0.71f, 1.0f))
    }

    "compute both eigen values and vectors" in {
      val (values, vectors) = matrix.round.eigen
      values.roundAt(2).eval shouldBe Tensor.vector(3f, 2f)
      vectors.roundAt(2).eval shouldBe
      Tensor.matrix(
        Array(0.71f, 0.0f),
        Array(0.71f, 1.0f))
    }
  }

  "QR factorization" should {
    "factor the matrix as Q and R" in {
      val input = Tensor.matrix(
        Array(0.1f, 0.0f),
        Array(0.5f, 1.0f),
        Array(0.7f, 2.0f))
      val (q, r) = input.const.qr()
      q.roundAt(6).eval shouldBe Tensor.matrix(
        Array(-0.11547f, -0.586353f),
        Array(-0.57735f, -0.617213f),
        Array(-0.80829f, 0.524631f))
      r.roundAt(6).eval shouldBe Tensor.matrix(
        Array(-0.866025f, -2.193931f),
        Array(0.0f, 0.432049f))
    }
  }

  "diagonal part" should {
    "extract diagonal tensor" in {
      val input = Tensor.matrix(
        Array(0.1f, 0.0f),
        Array(0.5f, 0.2f))
      input.const.diagPart.eval shouldBe Tensor.vector(0.1f, 0.2f)
    }
  }
}
