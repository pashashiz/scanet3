package scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import scanet.math.syntax._
import scanet.core.{Params, Shape, Tensor}
import scanet.models.Initializer
import scanet.optimizers.Adamax.{Momentum1, Momentum2}
import scanet.models.LinearRegression
import scanet.models.Loss._
import scanet.optimizers.Effect.RecordLoss
import scanet.optimizers.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

class AdamaxSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "Adamax" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = ds
      .train(LinearRegression())
      .loss(MeanSquaredError)
      .using(Adamax(rate = 0.1f))
      .batch(97)
      .each(1.epochs, RecordLoss())
      .stopAfter(100.epochs)
      .run()
    val loss = trained.loss.compile
    val TRecord(x, y) = ds.firstTensor(97)
    // same as Adam, the update rule itself is checked by the single step tests below
    loss(x, y).toScalar should be <= 12f
  }

  "Adamax" should "use the infinity norm of the gradient" in {
    // m = 0.1 * -2 = -0.2, m^ = m / (1 - 0.9) = -2, u = max(0.999 * 0, |-2|) = 2
    // delta = rate * m^ / u = 0.001 * -2 / 2
    val zero = Tensor.scalar(0f).const
    val Delta(delta, next) = Adamax(rate = 0.001f)
      .build[Float]((-2f).const, Params(Momentum1 -> zero, Momentum2 -> zero), 1.const)
    delta.eval should beCloseTo(Tensor.scalar(-0.001f))
    next(Momentum2).eval should beCloseTo(Tensor.scalar(2f))
  }

  "Adamax" should "decay the infinity norm when the gradient is smaller" in {
    // u = max(0.999 * 3, |1|) = 2.997
    val Delta(_, next) = Adamax()
      .build[Float](1f.const, Params(Momentum1 -> 0f.const, Momentum2 -> 3f.const), 2.const)
    next(Momentum2).eval should beCloseTo(Tensor.scalar(2.997f))
  }

  "Adamax" should "start with zero moments" in {
    Adamax().params(Shape(2)).values.map(_.initializer).toSet shouldBe Set(Initializer.Zeros)
  }
}
