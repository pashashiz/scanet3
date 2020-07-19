package org.scanet.models.nn

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.{OutputSeq, Tensor}
import org.scanet.estimators.accuracy
import org.scanet.models.{BinaryCrossentropy, LogisticRegression, MeanSquaredError, Sigmoid}
import org.scanet.optimizers.Effect.logResult
import org.scanet.optimizers.syntax._
import org.scanet.optimizers.{Adam, Tensor2Iterator}
import org.scanet.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

class ComposedLayer extends AnyFlatSpec with CustomMatchers  with SharedSpark with Datasets {

  "layers composition" should "produce right graph" in {
    val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
    val x = Tensor.matrix(
      Array(0f, 0f, 1f),
      Array(0f, 1f, 1f),
      Array(1f, 0f, 1f),
      Array(1f, 1f, 1f))
    val y = Tensor.matrix(Array(1f), Array(1f), Array(1f), Array(1f))
    val w1 = Tensor.matrix(
      Array(0f, 1f, 0.1f, 1f),
      Array(0f, 0.5f, 1f, 0f),
      Array(0f, 1f, 1f, 0.2f),
      Array(0f, 0.1f, 1f, 0.3f)).const
    val w2 = Tensor.matrix(
      Array(0f, 0.1f, 0.5f, 1f, 0f)).const
    val loss = model.withLoss(BinaryCrossentropy).build[Float](x.const, y.const, Seq(w1, w2))
    val grad: OutputSeq[Float] = loss.grad(Seq(w1, w2)).returns[Float]
      .zipWithIndex.map {case (grad, i) => grad.as(s"grad_$i")}
    grad.display()
  }

  it should "minimize logistic regression" in {
    val ds = logisticRegression.map(a => Array(a(0)/100, a(1)/100, a(2)))
    val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
    val trained = ds.train(model)
      .loss(BinaryCrossentropy)
      .using(Adam(0.1f))
      .initWith(s => Tensor.zeros(s))
      .batch(100)
      .each(1.epochs, logResult())
      .stopAfter(100.epochs)
      .run()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 100).next()
    val loss = trained.loss.compile()
    loss(x, y).toScalar should be <= 0.4f
    accuracy(trained, ds) should be >= 0.9f
    val predictor = trained.result.compile()
    val input = Tensor.matrix(
      Array(0.3462f, 0.7802f),
      Array(0.6018f, 0.8630f),
    )
    predictor(input).const.round.eval should be(Tensor.matrix(Array(0f), Array(1f)))
  }
}
