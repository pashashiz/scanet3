package org.scanet.optimizers

import org.scanet.core.Session.withing
import org.scanet.core._
import org.scanet.datasets.{Dataset, Iterator}
import org.scanet.math.{Dist, Numeric}
import org.scanet.math.syntax._
import org.scanet.models.Model
import org.scanet.optimizers.Condition.always
import org.scanet.optimizers.Optimizer.BuilderState._

import scala.annotation.tailrec

case class Step[W: Numeric: TensorType, R: Numeric: TensorType](
    size: Int, iter: Int = 1, epoch: Int = 1, result: () => Tensor[R] = null) {
  def nextIter: Step[W, R] = copy(iter = iter + 1)
  def nextEpoch: Step[W, R] = copy(epoch = epoch + 1, iter = 1)
  def total: Int = (epoch - 1) * size + iter
  def isLastIter: Boolean = iter == size
}

case class Optimizer[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](
      alg: Algorithm,
      model: Model[X, W, R],
      initArgs: Shape => Tensor[W],
      dataset: Dataset[X],
      batchSize: Int,
      minimizing: Boolean,
      stop: Condition[W, R],
      doOnEach: Effects[Step[W, R]]) {

  def run(): Tensor[W] = withing(session => {
    val result = model.result compile session
    val weightsAndMeta = TF2.identity[Float, Int].compose(model.grad) {
      case ((meta, iter), (w, g)) =>
        val Delta(delta, nextMeta) = alg.delta(g, meta, iter)
        val d = delta.cast[W]
        (if (minimizing) w - d else w + d, nextMeta)
    }.into[(Tensor[W], Tensor[Float])] compile session

    @tailrec
    def optimize(step: Step[W, R], effectState: Seq[_], it: Iterator[X], weights: Tensor[W], meta: Tensor[Float]): Tensor[W] = {
      if (!it.hasNext) {
        optimize(step.nextEpoch, effectState, dataset.iterator, weights, meta)
      } else {
        val batch = it.next(batchSize)
        val (nextWeight, nextMeta) = weightsAndMeta(meta, Tensor.scalar(step.total), batch, weights)
        val nextStep: Step[W, R] = step.copy(result = () => result(batch, nextWeight))
        val nextEffectState = doOnEach.action(effectState, nextStep)
        if (stop(step)) {
          nextWeight
        } else {
          optimize(nextStep.nextIter, nextEffectState, it, nextWeight, nextMeta)
        }
      }
    }
    val it = dataset.iterator
    optimize(Step(it.batches(batchSize)), doOnEach.unit, it, initArgs(it.shape), alg.initMeta(it.shape))
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
      copy(optimizer = optimizer.copy(initArgs = _ => args))

    def initWith(args: Shape => Tensor[W]): Builder[X, W, R, State] =
      copy(optimizer = optimizer.copy(initArgs = args))

    def on(dataset: Dataset[X]): Builder[X, W, R, State with WithDataset] =
      copy(optimizer = optimizer.copy(dataset = dataset))

    def stopWhen(condition: Condition[W, R]): Builder[X, W, R, State with WithStopCondition] =
      copy(optimizer = optimizer.copy(stop = condition))

    def stopAfter(condition: Condition[W, R]): Builder[X, W, R, State with WithStopCondition] =
      stopWhen(condition)

    def epochs(number: Int): Builder[X, W, R, State with WithStopCondition] =
      stopWhen(Condition.epochs(number))

    def iterations(number: Int): Builder[X, W, R, State with WithStopCondition] =
      stopWhen(Condition.iterations(number))

    def batch(size: Int): Builder[X, W, R, State] =
      copy(optimizer = optimizer.copy(batchSize = size))

    // NOTE: figure out how to rename it to each without getting issues
    // when type inference is required
    def doEach(when: Condition[W, R], action: Step[W, R] => Unit): Builder[X, W, R, State] =
      each(when, Effect.stateless[Step[W, R]](action))

    def doEach(action: Step[W, R] => Unit): Builder[X, W, R, State] =
      each(always, Effect.stateless[Step[W, R]](action))

    def each(effect: Effect[_, Step[W, R]]): Builder[X, W, R, State] =
      each(always, effect)

    def each(when: Condition[W, R], effect: Effect[_, Step[W, R]]): Builder[X, W, R, State] = {
      copy(optimizer = optimizer.copy(doOnEach = optimizer.doOnEach :+ effect.conditional(when)))
    }

    def build(implicit ev: State =:= Complete): Optimizer[X, W, R] = optimizer
  }

  def minimize[X: Numeric: TensorType, W: Numeric: TensorType: Dist, R: Numeric: TensorType](model: Model[X, W, R]): Builder[X, W, R, WithFunc] =
    Builder(Optimizer(null, model, s => Tensor.rand(s), null, Int.MaxValue, minimizing = true, always, Effects.empty))

  def maximize[X: Numeric: TensorType, W: Numeric: TensorType: Dist, R: Numeric: TensorType](model: Model[X, W, R]): Builder[X, W, R, WithFunc] =
    Builder(Optimizer(null, model, s => Tensor.rand(s), null, Int.MaxValue, minimizing = false, always, Effects.empty))
}
