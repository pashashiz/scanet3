package org.scanet.optimizers

import org.scanet.core.{Output, Tensor, TensorType}
import org.scanet.datasets.Dataset
import org.scanet.math.Numeric
import org.scanet.math.syntax._
import org.scanet.models.Model
import org.scanet.optimizers.Optimizer.BuilderState._

case class Step[W: Numeric: TensorType, R: Numeric: TensorType](
      iter: Int = 0, epoch: Int = 0, delta: Output[W] = null, result: Output[R] = null) {
  def isFirst: Boolean = iter == 0
  def nextIter: Step[W, R] = copy(iter = iter + 1)
  def nextEpoch: Step[W, R] = copy(epoch = epoch + 1)
}

case class Optimizer[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType, S](
      alg: Algorithm[S],
      model: Model[X, W, R],
      initArgs: Tensor[W],
      dataset: Dataset[X],
      batch: Int,
      minimizing: Boolean,
      stopCondition: Step[W, R] => Boolean,
      doOnEach: Step[W, R] => Unit) {

  def run(): Tensor[W] = {
    var step = Step[W, R]()
    var arg = initArgs
    var it = dataset.iterator
    var state = alg.initState(arg)
    while (step.isFirst || !stopCondition(step)) {
      if (it.hasNext) {
        val func = model(it.next(batch).const)
        val (delta, nextState) = alg.delta(func, arg.const, state)
        state = nextState
        arg = (if (minimizing) arg.const - delta.cast[W] else arg.const + delta.cast[W]).eval
        step = step.nextIter
          .copy(delta = delta.cast[W], result = func(arg.const))
        doOnEach(step)
      } else {
        it = dataset.iterator
        step = step.nextEpoch
      }
    }
    arg
  }
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

    def using[SS](alg: Algorithm[SS]): Builder[X, W, R, SS, State with WithAlg] =
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
      copy(optimizer = optimizer.copy(batch = size))

    def doOnEach(effect: Step[W, R] => Unit): Builder[X, W, R, S, State] =
      copy(optimizer = optimizer.copy(doOnEach = effect))

    def build(implicit ev: State =:= Complete): Optimizer[X, W, R, S] = optimizer
  }

  def minimize[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](model: Model[X, W, R]): Builder[X, W, R, _, WithFunc] =
    Builder(Optimizer(null, model, null, null, Int.MaxValue, minimizing = true, null, _ => ()))

  def maximize[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](model: Model[X, W, R]): Builder[X, W, R, _, WithFunc] =
    Builder(Optimizer(null, model, null, null, Int.MaxValue, minimizing = false, null, _ => ()))
}
