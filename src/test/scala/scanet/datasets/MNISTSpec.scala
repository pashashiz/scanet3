package scanet.datasets

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scanet.core.{Shape, Tensor, TensorBoard}
import scanet.images.Grayscale
import scanet.test.SharedSpark
import scanet.core.syntax._

import scala.collection.mutable

class MNISTSpec extends AnyFlatSpec with Matchers with SharedSpark {

  "MNIST training set" should "be downloaded" in {
    val set = MNIST.loadTrainingSet(1)(spark)
    val image = set.first().features
    renderImageAsASCII(image) should be
    """____________________________
      |____________________________
      |____________________________
      |____________________________
      |____________________________
      |____________xxxxxxxxxxxx____
      |________xxxxxxxxxxxxxxxx____
      |_______xxxxxxxxxxxxxxxx_____
      |_______xxxxxxxxxxx__________
      |________xxxxxxx_xx__________
      |_________xxxxx______________
      |___________xxxx_____________
      |___________xxxx_____________
      |____________xxxxxx__________
      |_____________xxxxxx_________
      |______________xxxxxx________
      |_______________xxxxx________
      |_________________xxxx_______
      |______________xxxxxxx_______
      |____________xxxxxxxx________
      |__________xxxxxxxxx_________
      |________xxxxxxxxxx__________
      |______xxxxxxxxxx____________
      |____xxxxxxxxxx______________
      |____xxxxxxxx________________
      |____________________________
      |____________________________
      |____________________________""".stripMargin
    val labels = set.first().labels
    labels should be(Array(0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f))
  }

  "10 MNIST images" should "be shown on a tensorboard" ignore {
    val set = MNIST.loadTrainingSet(10)(spark)
    set
      .collect()
      .foldLeft(TensorBoard("board"))((board, image) => {
        val label = image.labels.takeWhile(_ == 0f).length
        val tensor = Tensor(image.features, Shape(28, 28, 1))
        board.addImage(s"image $label", tensor, Grayscale())
      })
  }

  def renderImageAsASCII(data: Array[Float]): String = {
    val acc = mutable.Buffer[String]()
    for {
      row <- 0 until 28
      column <- 0 until 28
    } {
      val pixel = data(row * 28 + column)
      if (pixel == 0f)
        acc += "_"
      else
        acc += "x"
      if (column == 27)
        acc += "\n"
    }
    acc.mkString("")
  }
}
