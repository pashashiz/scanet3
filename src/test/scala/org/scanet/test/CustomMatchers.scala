package org.scanet.test

import org.scalatest.matchers.{MatchResult, Matcher}
import org.scalatest.matchers.should.Matchers
import org.scanet.linalg.{Shape, Tensor}

trait CustomMatchers extends Matchers {

  def haveShape(shape: Shape): Matcher[Tensor[_]] =
    tensor => {
      MatchResult(tensor.view.shape == shape, s"${tensor.view} was not equal to $shape", "", Vector(tensor, shape))
    }

  def containData[A](data: Array[A]): Matcher[Tensor[A]] =
    tensor => {
      val existing = tensor.toArray
      MatchResult(existing sameElements data, s"data $existing was not equal to data $data", "", Vector(existing, data))
    }
}
