package scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import scanet.math.syntax._
import scanet.core.{Params, Shape, Tensor}
import scanet.models.Initializer
import scanet.optimizers.Nadam.{Momentum, Velocity}
import scanet.models.LinearRegression
import scanet.models.Loss._
import scanet.optimizers.Effect.RecordLoss
import scanet.optimizers.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

class NadamSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "Nadam" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = ds
      .train(LinearRegression())
      .loss(MeanSquaredError)
      .using(Nadam(rate = 0.1f))
      .batch(97)
      .each(1.epochs, RecordLoss())
      .stopAfter(100.epochs)
      .run()
    val loss = trained.loss.compile
    val TRecord(x, y) = ds.firstTensor(97)
    // same as Adam, the update rule itself is checked by the single step tests below
    loss(x, y).toScalar should be <= 12f
  }

  "Nadam" should "scale the first step by the learning rate" in {
    // m = 0.1 * 2 = 0.2, v = 0.001 * 4 = 0.004
    // m^ = m / (1 - 0.9) = 2, g^ = g / (1 - 0.9) = 20, v^ = v / (1 - 0.999) = 4
    // delta = rate * (0.9 * m^ + 0.1 * g^) / sqrt(v^) = 0.001 * 3.8 / 2
    val zero = Tensor.scalar(0f).const
    val Delta(delta, next) = Nadam(rate = 0.001f)
      .build[Float](2f.const, Params(Momentum -> zero, Velocity -> zero), 1.const)
    delta.eval should beCloseTo(Tensor.scalar(0.0019f))
    next(Momentum).eval should beCloseTo(Tensor.scalar(0.2f))
    next(Velocity).eval should beCloseTo(Tensor.scalar(0.004f))
  }

  "Nadam" should "start with zero moments" in {
    Nadam().params(Shape(2)).values.map(_.initializer).toSet shouldBe Set(Initializer.Zeros)
  }
}
