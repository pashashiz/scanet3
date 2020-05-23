package org.scanet.optimizers

import org.scanet.core.Session.using
import org.scanet.core._
import org.scanet.datasets.Dataset
import org.scanet.datasets.Iterator
import org.scanet.math.{Convertible, Numeric}
import org.scanet.math.syntax._
import org.scanet.models.Model
import org.scanet.optimizers.Optimizer.BuilderState._
import org.scanet.optimizers.Optimizer.Condition

import scala.annotation.tailrec

case class Step[W: Numeric: TensorType, R: Numeric: TensorType](
      iter: Int = 0, epoch: Int = 0, result: () => Tensor[R] = null) {
  def isFirst: Boolean = iter == 0
  def nextIter: Step[W, R] = copy(iter = iter + 1)
  def nextEpoch: Step[W, R] = copy(epoch = epoch + 1)
}

case class Effect[A, S](acc: A, action: (A, S) => A)

object Effect {

  def plotResult[W: Numeric: TensorType, R: Numeric: TensorType]
    (name: String = "result", dir: String = "board")
    (implicit c: Convertible[R, Float]): Effect[TensorBoard, Step[W, R]] = {
    Effect(new TensorBoard(dir), (board, step) => {
      board.addScalar(name, step.result().toScalar, step.iter)
      board
    })
  }

  def logResult[W: Numeric: TensorType, R: Numeric: TensorType]()
    (implicit c: Convertible[R, Float]): Effect[Unit, Step[W, R]] = {
    Effect(null, (_, step) => {
      println(s"#${step.epoch}:${step.iter} ${step.result().toScalar}")
    })
  }
}

case class Effects[S](all: Seq[Effect[_, S]]) {
  def acc: Seq[_] = all.map(_.acc)
  def action(acc: Seq[_], next: S): Seq[_] = acc.zip(all) map {
    case (a, Effect(_, action)) => action(a, next)
  }
  def :+ (effect: Effect[_, S]): Effects[S] = Effects(all :+ effect)
}

object Effects {

  def empty[S]: Effects[S] = Effects(Seq())
}

object Conditions {

  def iterations[W: Numeric: TensorType, R: Numeric: TensorType](number: Int): Condition[W, R] = {
    step => step.iter % number == 0
  }

  def epochs[W: Numeric: TensorType, R: Numeric: TensorType](number: Int): Condition[W, R] = {
    step => step.epoch % number == 0
  }
}

case class Optimizer[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](
      alg: Algorithm,
      model: Model[X, W, R],
      initArgs: Tensor[W],
      dataset: Dataset[X],
      batchSize: Int,
      minimizing: Boolean,
      stop: Condition[W, R],
      doOnEach: Effects[Step[W, R]]) {

  def run(): Tensor[W] = using(session => {
    val result = model.result compile session
    val weightsTF: TF3[Float, X, W, (Output[W], Output[Float]), (Tensor[W], Tensor[Float])] =
      TF1.identity[Float].compose(model.grad) {
        case (meta, (w, g)) =>
          val Delta(delta, nextMeta) = alg.delta(g, meta)
          val d = delta.cast[W]
          (if (minimizing) w - d else w + d, nextMeta)
      }
    val weightsAndMeta = weightsTF compile session

    @tailrec
    def optimize(step: Step[W, R], effectState: Seq[_], it: Iterator[X], weights: Tensor[W], meta: Tensor[Float]): Tensor[W] = {
      if (stop(step)) weights
      else if (!it.hasNext) optimize(step.nextEpoch, effectState, dataset.iterator, weights, meta)
      else {
        val batch = it.next(batchSize)
        val (nextWeight, nextMeta) = weightsAndMeta(meta, batch, weights)
        val nextStep: Step[W, R] = step.copy(result = () => result(batch, nextWeight))
        val nextEffectState = doOnEach.action(effectState, nextStep)
        optimize(nextStep.nextIter, nextEffectState, it, nextWeight, nextMeta)
      }
    }

    optimize(Step(), doOnEach.acc, dataset.iterator, initArgs, alg.initMeta(initArgs))
  })
}

object Optimizer {

  type Condition[W, R] = (Step[W, R] => Boolean)

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

    def stopWhen(condition: Condition[W, R]): Builder[X, W, R, State with WithStopCondition] =
      copy(optimizer = optimizer.copy(stop = condition))

    def epochs(number: Int): Builder[X, W, R, State with WithStopCondition] =
      stopWhen(step => step.epoch == number)

    def iterations(number: Int): Builder[X, W, R, State with WithStopCondition] =
      stopWhen(step => step.iter == number)

    def batch(size: Int): Builder[X, W, R, State] =
      copy(optimizer = optimizer.copy(batchSize = size))

    def each(where: Condition[W, R], action: Step[W, R] => Unit): Builder[X, W, R, State] =
      each(where, Effect[Any, Step[W, R]](null, (_, step) => action(step)))

    def each(action: Step[W, R] => Unit): Builder[X, W, R, State] =
      each(_ => true, Effect[Any, Step[W, R]](null, (_, step) => action(step)))

    def each[S](effect: Effect[S, Step[W, R]]): Builder[X, W, R, State] =
      each(_ => true, effect)

    def each[S](where: Condition[W, R], effect: Effect[S, Step[W, R]]): Builder[X, W, R, State] = {
      val conditionalEffect = effect.copy(action = (state: S, step: Step[W, R]) => {
        if (where(step)) {
          effect.action(state, step)
        } else {
          state
        }
      })
      copy(optimizer = optimizer.copy(doOnEach = optimizer.doOnEach :+ conditionalEffect))
    }

    def build(implicit ev: State =:= Complete): Optimizer[X, W, R] = optimizer
  }

  def minimize[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](model: Model[X, W, R]): Builder[X, W, R, WithFunc] =
    Builder(Optimizer(null, model, null, null, Int.MaxValue, minimizing = true, step => step.iter > 0, Effects.empty))

  def maximize[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](model: Model[X, W, R]): Builder[X, W, R, WithFunc] =
    Builder(Optimizer(null, model, null, null, Int.MaxValue, minimizing = false, step => step.iter > 0, Effects.empty))
}
