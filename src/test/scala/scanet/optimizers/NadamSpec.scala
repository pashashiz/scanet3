package scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import scanet.core.Tensor
import scanet.math.syntax._
import scanet.models.LinearRegression
import scanet.models.Loss._
import scanet.optimizers.Effect.RecordLoss
import scanet.optimizers.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

class NadamSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "Nadam" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = ds
      .train(LinearRegression)
      .loss(MeanSquaredError)
      .using(Nadam())
      .initWith(Tensor.zeros(_))
      .batch(97)
      .each(1.epochs, RecordLoss())
      .stopAfter(50.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 97).next()
    loss(x, y).toScalar should be <= 9f
  }
}
