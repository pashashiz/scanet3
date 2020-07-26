package org.scanet.datasets

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.test.SharedSpark

class MNISTSpec extends AnyFlatSpec with Matchers with SharedSpark {

  "MNIST training set" should "be downloaded" in {
    val set = MNIST.loadTrainingSet(sc, 5)
    set.collect().foreach(v => printImage(v))
  }

  def printImage(data: Array[Float]): Unit = {
    for {
      row <- 0 until 28
      column <- 0 until 28
    } {
      val pixel = data(row * 28 + column)
      if (pixel == 0f)
        print("_")
      else
        print("x")
      if (column == 27)
        println()
    }
    data.slice(784, 794).foreach(print(_))
    println()
  }
}
