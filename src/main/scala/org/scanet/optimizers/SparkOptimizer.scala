package org.scanet.optimizers

import org.apache.spark.rdd.RDD
import org.scanet.core.Session.withing
import org.scanet.core.{Tensor, _}
import org.scanet.math.syntax._
import org.scanet.math.{Dist, Numeric}
import org.scanet.models.Model
import org.scanet.optimizers.Condition.always
import org.scanet.optimizers.SparkOptimizer.BuilderState._

import scala.annotation.{meta, tailrec}


case class SparkOptimizer[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](
      alg: Algorithm,
      model: Model[X, W, R],
      initArgs: Shape => Tensor[W],
      dataset: RDD[Array[X]],
      batchSize: Int, // todo: set new default batch size
      minimizing: Boolean,
      stop: Condition[W, R],
      doOnEach: Effects[Step[W, R]]) {

  def run(): Tensor[W] = {
    val ds = dataset.repartition(2).cache() // todo: set partition

    val features = ds.first().length // todo: figure that out before
    val sc = ds.sparkContext

    @tailrec
    def optimize(prevStep: Step[W, R], weights: Tensor[W], meta: Tensor[Float]): Tensor[W] = {
      val weightsBr = sc.broadcast(weights)
      val metaBr = sc.broadcast(meta)
      val (iter, newWeights, newMeta) = ds
        .mapPartitions(it => Iterator(optimizeOnPartition(
          it, prevStep.iter, features, weightsBr.value, metaBr.value)))
        .treeReduce(averageMetaAndWeights)
      println("REDUCED")
      val step = prevStep.nextEpoch.incIter(iter)
      println(step) // todo: apply side effects
      if (stop(step)) {
        newWeights
      } else {
        optimize(step, newWeights, newMeta)
      }
    }

    val weightsShape = model.shape(features)
    optimize(Step[W, R](), initArgs(weightsShape), alg.initMeta(weightsShape))
  }

  private def optimizeOnPartition(
        it: scala.Iterator[Array[X]], globalIter: Int, features: Int,
        weights: Tensor[W], meta: Tensor[Float]): (Int, Tensor[W], Tensor[Float]) = {
    val result = withing(session => {
      // todo: cache TF
      val calc = TF2.identity[Float, Int].compose(model.grad) {
        case ((meta, iter), (w, g)) =>
          val Delta(delta, nextMeta) = alg.delta(g, meta, iter)
          val d = delta.cast[W]
          (if (minimizing) w - d else w + d, nextMeta)
      }.into[(Tensor[W], Tensor[Float])] compile session
      val batches: BatchingIterator[X] = BatchingIterator(it, batchSize, features)

      @tailrec
      def optimize(iter: Int, weights: Tensor[W], meta: Tensor[Float]): (Int, Tensor[W], Tensor[Float]) = {
        println(s"OPTIMIZING $iter:")
        println(s"WEIGHTS: $weights")
        println(s"META: $meta")
        if (batches.hasNext) {
          val (nextWeight, nextMeta) = calc(meta, Tensor.scalar(globalIter + iter), batches.next(), weights)
          optimize(iter + 1, nextWeight, nextMeta)
        } else {
          (iter, weights, meta)
        }
      }

      optimize(0, weights, meta)
    })
    result
  }

  private def averageMetaAndWeights(
       left: (Int, Tensor[W], Tensor[Float]),
       right: (Int, Tensor[W], Tensor[Float])): (Int, Tensor[W], Tensor[Float]) = {
    // todo: cache TF
    val (leftIter, leftWeights, leftMeta) = left
    val (rightIter, rightWeights, rightMeta) = right
    val avgWeights = ((leftWeights.const + rightWeights.const) / 2.0f.const.cast[W]).eval
    val avfMeta = ((leftMeta.const + rightMeta.const) / 2.0f.const).eval
    (leftIter + rightIter, avgWeights, avfMeta)
  }
}

object SparkOptimizer {

  sealed trait BuilderState

  object BuilderState {
    sealed trait WithAlg extends BuilderState
    sealed trait WithFunc extends BuilderState
    sealed trait WithDataset extends BuilderState
    sealed trait WithStopCondition extends BuilderState
    type Complete = WithAlg with WithFunc with WithDataset with WithStopCondition
  }

  case class Builder[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType, State <: BuilderState](optimizer: SparkOptimizer[X, W, R]) {

    def using(alg: Algorithm): Builder[X, W, R, State with WithAlg] =
      copy(optimizer = optimizer.copy(alg = alg))

    def initWith(args: Tensor[W]): Builder[X, W, R, State] =
      copy(optimizer = optimizer.copy(initArgs = _ => args))

    def initWith(args: Shape => Tensor[W]): Builder[X, W, R, State] =
      copy(optimizer = optimizer.copy(initArgs = args))

    def on(dataset: RDD[Array[X]]): Builder[X, W, R, State with WithDataset] =
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

    def build(implicit ev: State =:= Complete): SparkOptimizer[X, W, R] = optimizer
  }

  def minimize[X: Numeric: TensorType, W: Numeric: TensorType: Dist, R: Numeric: TensorType](model: Model[X, W, R]): Builder[X, W, R, WithFunc] =
    Builder(SparkOptimizer(null, model, s => Tensor.rand(s), null, Int.MaxValue, minimizing = true, always, Effects.empty))

  def maximize[X: Numeric: TensorType, W: Numeric: TensorType: Dist, R: Numeric: TensorType](model: Model[X, W, R]): Builder[X, W, R, WithFunc] =
    Builder(SparkOptimizer(null, model, s => Tensor.rand(s), null, Int.MaxValue, minimizing = false, always, Effects.empty))
}
