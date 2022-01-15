package scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import scanet.core.Tensor
import scanet.math.syntax._
import scanet.models.LinearRegression
import scanet.models.Loss._
import scanet.optimizers.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

class AdaGradSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "AdaGrad" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = ds
      .train(LinearRegression())
      .loss(MeanSquaredError)
      .using(AdaGrad(rate = 1f))
      .initWith(Tensor.zeros(_))
      .batch(97)
      .stopAfter(100.epochs)
      .run()
    val loss = trained.loss.compile()
    val TRecord(x, y) = ds.firstTensor(97)
    loss(x, y).toScalar should be <= 9f
  }
}
