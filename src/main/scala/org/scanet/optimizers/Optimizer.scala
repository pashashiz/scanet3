package org.scanet.optimizers

import org.scanet.core.Session.using
import org.scanet.core.{Output, Tensor, TensorType}
import org.scanet.datasets.Dataset
import org.scanet.math.Numeric
import org.scanet.math.syntax._
import org.scanet.models.Model
import org.scanet.optimizers.Optimizer.BuilderState._

case class Step[W: Numeric: TensorType, R: Numeric: TensorType](
      iter: Int = 0, epoch: Int = 0, result: Output[R] = null) {
  def isFirst: Boolean = iter == 0
  def nextIter: Step[W, R] = copy(iter = iter + 1)
  def nextEpoch: Step[W, R] = copy(epoch = epoch + 1)
}

case class Optimizer[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](
      alg: Algorithm,
      model: Model[X, W, R],
      initArgs: Tensor[W],
      dataset: Dataset[X],
      batchSize: Int,
      minimizing: Boolean,
      stopCondition: Step[W, R] => Boolean,
      doOnEach: Step[W, R] => Unit) {

  def run(): Tensor[W] = using(session => {
    var step = Step[W, R]()
    var it = dataset.iterator
    var weights = initArgs
    // input placeholders
    val batchPl = placeholder[X](it.shape(batchSize))
    val weightPl = placeholder[W](initArgs.shape)
    // graph of compiled and optimized model
    val compiled = model(batchPl, weightPl)
    var Delta(delta, metadata) = alg.delta(compiled, weightPl)
    val optimizedWeights = if (minimizing) weightPl - delta.cast[W] else weightPl + delta.cast[W]
    // iterate model in session
    while (step.isFirst || !stopCondition(step)) {
      if (it.hasNext) {
        val batch = it.next(batchSize)
        val (newWeight, newMeta)  = session.runner
          .feed(batchPl -> batch, weightPl -> weights)
          .feed(metadata.feed)
          .eval(optimizedWeights, metadata.outputs)
        weights = newWeight
        metadata = metadata.next(newMeta)

        step = step.nextIter.copy(result = model(batch.const, weights.const))
        doOnEach(step)
      } else {
        it = dataset.iterator
        step = step.nextEpoch
      }
    }
    weights
  })
}

object Optimizer {

  sealed trait BuilderState

  object BuilderState {
    sealed trait WithAlg extends BuilderState
    sealed trait WithFunc extends BuilderState
    sealed trait WithDataset extends BuilderState
    sealed trait WithStopCondition extends BuilderState
    type Complete = WithAlg with WithFunc with WithDataset with WithStopCondition
  }

  case class Builder[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType, State <: BuilderState](optimizer: Optimizer[X, W, R]) {

    def using(alg: Algorithm): Builder[X, W, R, State with WithAlg] =
      copy(optimizer = optimizer.copy(alg = alg))

    def initWith(args: Tensor[W]): Builder[X, W, R, State] =
      copy(optimizer = optimizer.copy(initArgs = args))

    def on(dataset: Dataset[X]): Builder[X, W, R, State with WithDataset] =
      copy(optimizer = optimizer.copy(dataset = dataset))

    def stopWhen(condition: Step[W, R] => Boolean): Builder[X, W, R, State with WithStopCondition] =
      copy(optimizer = optimizer.copy(stopCondition = condition))

    def epochs(number: Int): Builder[X, W, R, State with WithStopCondition] =
      stopWhen(step => step.epoch == number)

    def iterations(number: Int): Builder[X, W, R, State with WithStopCondition] =
      stopWhen(step => step.iter == number)

    def batch(size: Int): Builder[X, W, R, State] =
      copy(optimizer = optimizer.copy(batchSize = size))

    def doOnEach(effect: Step[W, R] => Unit): Builder[X, W, R, State] =
      copy(optimizer = optimizer.copy(doOnEach = effect))

    def build(implicit ev: State =:= Complete): Optimizer[X, W, R] = optimizer
  }

  def minimize[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](model: Model[X, W, R]): Builder[X, W, R, WithFunc] =
    Builder(Optimizer(null, model, null, null, Int.MaxValue, minimizing = true, null, _ => ()))

  def maximize[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](model: Model[X, W, R]): Builder[X, W, R, WithFunc] =
    Builder(Optimizer(null, model, null, null, Int.MaxValue, minimizing = false, null, _ => ()))
}
