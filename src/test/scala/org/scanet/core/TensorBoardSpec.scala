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

  "computation graph" should "be displayed 2" in {
    val c = (1.0f.const.as("a") * 2.0f.const.as("a")).as("c")
    val d = (c plus 10.0f.const minus 5.0f.const).as("d")
    d.display()
  }
}
