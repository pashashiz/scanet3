package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.math.syntax._

import scala.reflect.io.Path._

class TensorBoardSpec extends AnyFlatSpec with Matchers {

  "computation graph" should "be displayed" in {
    val a = 1.0f.const.as("a")
    val b = 1.0f.const.as("b")
    val c = (a plus b).as("c")
    c.display("tmp")
    // run tensor-board: tensorboard --logdir tmp
    val files = "tmp".toDirectory.files.map(_.path).filter(_ matches ".*events.out.tfevents.*")
    files should not be empty
    "tmp".toDirectory.deleteRecursively()
  }

  "scalar" should "be displayed" in {
    new TensorBoard().addScalar("a", 11.0f, 0)
  }
}
