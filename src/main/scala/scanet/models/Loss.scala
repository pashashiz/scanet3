package scanet.models

import scanet.core.{Expr, Floating}
import scanet.math.syntax._

trait Loss {
  def build[A: Floating](predicted: Expr[A], expected: Expr[A]): Expr[A]
}

object Loss {

  case object Identity extends Loss {
    override def build[A: Floating](
        predicted: Expr[A],
        expected: Expr[A]): Expr[A] = predicted
  }

  case object MeanSquaredError extends Loss {
    override def build[A: Floating](
        predicted: Expr[A],
        expected: Expr[A]): Expr[A] = {
      (predicted - expected).sqr.mean
    }
  }

  case object BinaryCrossentropy extends Loss {
    override def build[A: Floating](
        predicted: Expr[A],
        expected: Expr[A]): Expr[A] = {
      val one = 1.0f.const.cast[A]
      val epsilon = 1e-8f.const.cast[A]
      // if we expect 1 and
      // - predicted 1 - then loss 0
      // - predicted 0 - then loss -indefinite (need epsilon here)
      val left = expected * (predicted + epsilon).log
      // if we expect 0 and
      // - predicted 0 - then loss 0
      // - predicted 1 - then loss -indefinite (need epsilon here)
      val right = (one - expected) * (one - (predicted - epsilon)).log
      (left.negate - right).mean
    }
  }

  case object CategoricalCrossentropy extends Loss {
    override def build[A: Floating](
        predicted: Expr[A],
        expected: Expr[A]): Expr[A] = {
      val epsilon = 1e-8f.const.cast[A]
      // if we expect 1 and
      // - predicted 1 - then loss 0
      // - predicted 0 - then loss -indefinite (need epsilon here)
      // if we expect 0
      // - ignore result
      val num = expected.shape.head.const.cast[A]
      (expected * (predicted + epsilon).log).sum.negate / num
    }
  }
}
