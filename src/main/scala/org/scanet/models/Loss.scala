package org.scanet.models

import org.scanet.core.{Output, TensorType}
import org.scanet.math.{Floating, Numeric}
import org.scanet.math.syntax._

trait Loss {
  def build[A: Numeric: Floating: TensorType]
  (predicted: Output[A], expected: Output[A]): Output[A]
}

case object Identity extends Loss {
  override def build[A: Numeric: Floating: TensorType]
  (predicted: Output[A], expected: Output[A]): Output[A] = predicted
}

case object MeanSquaredError extends Loss {
  override def build[A: Numeric: Floating: TensorType]
  (predicted: Output[A], expected: Output[A]): Output[A] = {
    (predicted - expected).sqr.mean
  }
}

case object BinaryCrossentropy extends Loss {
  override def build[A: Numeric: Floating: TensorType]
  (predicted: Output[A], expected: Output[A]): Output[A] = {
    val one = 1.0f.const.cast[A]
    val left = expected :* predicted.log
    val right = (one - expected) :* (one - predicted).log
    (left.negate - right).mean
  }
}