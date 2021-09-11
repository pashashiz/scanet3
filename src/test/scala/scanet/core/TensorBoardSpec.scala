package scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import scanet.images.Grayscale
import scanet.math.syntax._
import scanet.test.CustomMatchers

import scala.reflect.io.Path._

class TensorBoardSpec extends AnyFlatSpec with CustomMatchers {

  // run tensor-board: tensorboard --logdir tmp
  "computation graph" should "be displayed" ignore {
    val a = 1.0f.const.as("a")
    val b = 1.0f.const.as("b")
    val c = (a plus b).as("c")
    c.display("tmp")
    "tmp" should haveTensorBoardFiles
    "tmp".toDirectory.deleteRecursively()
  }

  "scalar" should "be displayed" ignore {
    TensorBoard("tmp").addScalar("a", 11.0f, 0)
    "tmp" should haveTensorBoardFiles
    "tmp".toDirectory.deleteRecursively()
  }

  "image" should "be displayed" in {
    val tensor = Tensor.matrix(Array(-1f, 1f, 1f), Array(0f, -0.3f, 0.3f)).reshape(2, 3, 1)
    TensorBoard("tmp").addImage("image", tensor, Grayscale())
    "tmp" should haveTensorBoardFiles
    "tmp".toDirectory.deleteRecursively()
  }
}
