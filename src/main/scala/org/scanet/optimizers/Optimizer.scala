package org.scanet.optimizers

import org.scanet.core.Session.using
import org.scanet.core._
import org.scanet.datasets.Dataset
import org.scanet.datasets.Iterator
import org.scanet.math.Numeric
import org.scanet.math.syntax._
import org.scanet.models.Model
import org.scanet.optimizers.Optimizer.BuilderState._

import scala.annotation.tailrec

case class Step[W: Numeric: TensorType, R: Numeric: TensorType](
      iter: Int = 0, epoch: Int = 0, result: () => Tensor[R] = null) {
  def isFirst: Boolean = iter == 0
  def nextIter(res: () => Tensor[R]): Step[W, R] = copy(iter = iter + 1, result = res)
  def nextEpoch: Step[W, R] = copy(epoch = epoch + 1)
}

case class Optimizer[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType, S](
      alg: Algorithm,
      model: Model[X, W, R],
      initArgs: Tensor[W],
      dataset: Dataset[X],
      batchSize: Int,
      minimizing: Boolean,
      stopCondition: Step[W, R] => Boolean,
      doOnEach: Step[W, R] => Unit) {

  def run(): Tensor[W] = using(session => {
    val result = model.result compile session
    val weightsAndMeta = TF1.identity[Float].compose(model.grad) {
      case (meta, (w, g)) =>
        val Delta(delta, nextMeta) = alg.delta(g, meta)
        val d = delta.cast[W]
        (if (minimizing) w - d else w + d, nextMeta)
    }.into[(Tensor[W], Tensor[Float])] compile session

    @tailrec
    def optimize(step: Step[W, R], it: Iterator[X], weights: Tensor[W], meta: Tensor[Float]): Tensor[W] = {
      if (stopCondition(step)) weights
      else if (!it.hasNext) optimize(step.nextEpoch, dataset.iterator, weights, meta)
      else {
        val batch = it.next(batchSize)
        val (nextWeight, nextMeta) = weightsAndMeta(meta, batch, weights)
        val nextStep = step.nextIter(() => result(batch, nextWeight))
        doOnEach(nextStep)
        optimize(nextStep, it, nextWeight, nextMeta)
      }
    }

    optimize(Step(), dataset.iterator, initArgs, alg.initMeta(initArgs))
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

  case class Builder[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType, S, State <: BuilderState](optimizer: Optimizer[X, W, R, S]) {

    def using[SS](alg: Algorithm): Builder[X, W, R, SS, State with WithAlg] =
      copy(optimizer = optimizer.copy(alg = alg))

    def initWith(args: Tensor[W]): Builder[X, W, R, S, State] =
      copy(optimizer = optimizer.copy(initArgs = args))

    def on(dataset: Dataset[X]): Builder[X, W, R, S, State with WithDataset] =
      copy(optimizer = optimizer.copy(dataset = dataset))

    def stopWhen(condition: Step[W, R] => Boolean): Builder[X, W, R, S, State with WithStopCondition] =
      copy(optimizer = optimizer.copy(stopCondition = condition))

    def epochs(number: Int): Builder[X, W, R, S, State with WithStopCondition] =
      stopWhen(step => step.epoch == number)

    def iterations(number: Int): Builder[X, W, R, S, State with WithStopCondition] =
      stopWhen(step => step.iter == number)

    def batch(size: Int): Builder[X, W, R, S, State] =
      copy(optimizer = optimizer.copy(batchSize = size))

    def doOnEach(effect: Step[W, R] => Unit): Builder[X, W, R, S, State] =
      copy(optimizer = optimizer.copy(doOnEach = effect))

    def build(implicit ev: State =:= Complete): Optimizer[X, W, R, S] = optimizer
  }

  def minimize[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](model: Model[X, W, R]): Builder[X, W, R, _, WithFunc] =
    Builder(Optimizer(null, model, null, null, Int.MaxValue, minimizing = true, step => step.iter > 0, _ => ()))

  def maximize[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](model: Model[X, W, R]): Builder[X, W, R, _, WithFunc] =
    Builder(Optimizer(null, model, null, null, Int.MaxValue, minimizing = false, step => step.iter > 0, _ => ()))
}
