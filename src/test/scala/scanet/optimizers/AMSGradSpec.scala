package scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import scanet.math.syntax._
import scanet.models.LinearRegression
import scanet.models.Loss._
import scanet.optimizers.Effect.RecordLoss
import scanet.optimizers.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

class AMSGradSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "AMSGrad" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = ds
      .train(LinearRegression())
      .loss(MeanSquaredError)
      .using(AMSGrad(rate = 0.1f))
      .batch(97)
      .each(1.epochs, RecordLoss())
      .stopAfter(100.epochs)
      .run()
    val loss = trained.loss.compile()
    val TRecord(x, y) = ds.firstTensor(97)
    loss(x, y).toScalar should be <= 9f
  }
}
