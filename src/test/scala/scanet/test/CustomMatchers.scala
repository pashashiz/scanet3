package scanet.test

import org.scalatest.matchers.should.Matchers
import org.scalatest.matchers.{MatchResult, Matcher}
import scanet.core.{Shape, Tensor}
import scala.reflect.io.Path._

trait CustomMatchers extends Matchers {

  def haveShape(shape: Shape): Matcher[Tensor[_]] =
    tensor => {
      MatchResult(
        tensor.view.shape == shape,
        s"${tensor.view} was not equal to $shape",
        "",
        Vector(tensor, shape))
    }

  def containData[A](data: Array[A]): Matcher[Tensor[A]] =
    tensor => {
      val existing = tensor.toArray
      MatchResult(
        existing sameElements data,
        s"data ${existing.mkString(", ")} was not equal to data ${data.mkString(", ")}",
        "",
        Vector(existing, data))
    }

  def beCloseTo(expected: Tensor[Float], tolerance: Float = 1e-6f): Matcher[Tensor[Float]] =
    tensor => {
      val actual = tensor.toArray
      val sameShape = tensor.view.shape == expected.view.shape
      MatchResult(
        sameShape && actual.zip(expected.toArray).forall { case (a, e) =>
          math.abs(a - e) <= tolerance
        },
        s"$tensor was not close to $expected with tolerance $tolerance",
        s"$tensor was close to $expected with tolerance $tolerance")
    }

  def beWithinTolerance(mean: Float, tolerance: Float): Matcher[Float] =
    be >= (mean - tolerance) and be <= (mean + tolerance)

  def haveTensorBoardFiles: Matcher[String] =
    dir => {
      MatchResult(
        dir.toDirectory.files.map(_.path).exists(_ matches ".*events.out.tfevents.*"),
        s"directory '$dir' does not contain tf events",
        "",
        Vector(dir))
    }
}
