package org.scanet.optimizers

import org.apache.spark.rdd.RDD
import org.scanet.core.{Tensor, _}
import org.scanet.math.syntax._
import org.scanet.math.{Convertible, Dist, Floating, Numeric}
import org.scanet.models.{Model, TrainedModel}
import org.scanet.optimizers.Condition.always
import org.scanet.optimizers.Optimizer.BuilderState._
import org.scanet.optimizers.Optimizer.{sessionsPool, tfCache}

import scala.annotation.tailrec
import scala.collection.mutable

case class Step[R: Numeric: TensorType](
    epoch: Int = 0, iter: Int = 0, result: Option[R] = None) {
  def nextIter: Step[R] = incIter(1)
  def incIter(number: Int): Step[R] = copy(iter = iter + number)
  def nextEpoch: Step[R] = copy(epoch = epoch + 1)
  def withResult(value: R): Step[R] = copy(result = Some(value))
  override def toString: String = s"$epoch:$iter"
}

// E - type of input dataset to train on, could have any numeric values
// R - type to use on a model, could be only Float or Double
case class Optimizer[
  E: Numeric : TensorType,
  R: Numeric : Floating : TensorType](
     alg: Algorithm,
     model: Model[E, R],
     initArgs: Shape => Tensor[R],
     dataset: RDD[Array[E]],
     partitons: Int,
     batchSize: Int,
     minimizing: Boolean,
     stop: Condition[R],
     @transient doOnEach: Effects[Step[R]])
(implicit c: Convertible[Int, R]) {

  def run(): TrainedModel[E, R] = {
    val ds = dataset.repartition(partitons).cache()
    val sc = ds.sparkContext

    @tailrec
    def optimize(prevStep: Step[R], effectState: Seq[_], weights: Tensor[R], meta: Tensor[R]): Tensor[R] = {
      val weightsBr = sc.broadcast(weights)
      val metaBr = sc.broadcast(meta)
      val (iter, newWeights, newMeta, result) = ds
        .mapPartitions(it => Iterator(optimizeOnPartition(
          it, prevStep.iter, weightsBr.value, metaBr.value)))
        .treeReduce(averageMetaAndWeights)
      val step: Step[R] = prevStep.nextEpoch.incIter(iter).withResult(result)
      val nextEffectState = doOnEach.action(effectState, step)
      if (stop(step)) {
        newWeights
      } else {
        optimize(step, nextEffectState, newWeights, newMeta)
      }
    }
    val weights = optimize(Step(), doOnEach.unit, Tensor.zeros(), Tensor.zeros())
    model.trained(weights)
  }

  private def optimizeOnPartition(
      it: scala.Iterator[Array[E]], globalIter: Int, weights: Tensor[R], meta: Tensor[R]): (Int, Tensor[R], Tensor[R], R) = {
    val result = sessionsPool.withing(session => {
      val batches = Tensor2Iterator(it, batchSize, splitAt = size => size - model.outputs())
      val (weightsInitialized, metaInitialized) = if (globalIter == 0) {
        val features = batches.columns - model.outputs()
        val shape = model.weightsShape(features)
        (initArgs(shape), alg.initMeta[R](shape))
      } else {
        (weights, meta)
      }

      val loss = tfCache.getOrCompute(s"$model:loss", model.loss) compile session
      val calc = tfCache.getOrCompute(s"$model:$alg:calc",
        model.weightsAndGrad.compose(TF2.identity[R, Int]) {
          case ((w, g), (meta, iter)) =>
            val Delta(del, nextMeta) = alg.delta[R](g, meta, iter)
            val d = del.cast[R]
            (if (minimizing) w - d else w + d, nextMeta)
        }.into[(Tensor[R], Tensor[R])]) compile session

      @tailrec
      def optimize(iter: Int, weights: Tensor[R], meta: Tensor[R]): (Int, Tensor[R], Tensor[R], R) = {
        val (x, y) = batches.next()
        val (nextWeights, nextMeta) = calc(x, y, weights, meta, Tensor.scalar(globalIter + iter + 1))
        if (batches.hasNext) {
          optimize(iter + 1, nextWeights, nextMeta)
        } else {
          (iter + 1, nextWeights, nextMeta, loss(x, y, nextWeights).toScalar)
        }
      }

      optimize(0, weightsInitialized, metaInitialized)
    })
    result
  }

  // todo: tuple3 -> case class
  private def averageMetaAndWeights(
                                     left: (Int, Tensor[R], Tensor[R], R),
                                     right: (Int, Tensor[R], Tensor[R], R)): (Int, Tensor[R], Tensor[R], R) = {
    sessionsPool.withing(session => {
      val (leftIter, leftWeights, leftMeta, leftResult) = left
      val (rightIter, rightWeights, rightMeta, rightResult) = right
      val weightsAvg = tfCache.getOrCompute("weightsAvg", avg[R]) compile session
      val metaAvg = tfCache.getOrCompute("metaAvg", avg[R]) compile session
      val resultAvg = (leftResult plus rightResult) / c.convert(2)
      (leftIter + rightIter, weightsAvg(leftWeights, rightWeights), metaAvg(leftMeta, rightMeta), resultAvg)
    })
  }

  private def avg[A: Numeric: TensorType]: TF2[A, A, Output[A], Tensor[A]] =
    TF2((arg1: Output[A], arg2: Output[A]) => (arg1 + arg2) / 2.0f.const.cast[A]).returns[Tensor[A]]
}

object Optimizer {

  class Cache {
    private val map = mutable.Map[String, Any]()
    def getOrCompute[A](key: String, op: => A): A = {
      map.get(key) match {
        case Some(v) => v.asInstanceOf[A]
        case None => val d = op; map(key) = d; d
      }
    }
  }

  val sessionsPool = new SessionPool(64)
  val tfCache = new Cache

  sealed trait BuilderState

  object BuilderState {
    sealed trait WithAlg extends BuilderState
    sealed trait WithFunc extends BuilderState
    sealed trait WithDataset extends BuilderState
    sealed trait WithStopCondition extends BuilderState
    type Complete = WithAlg with WithFunc with WithDataset with WithStopCondition
  }

  case class Builder[
    X: Numeric: TensorType,
    R: Numeric: Floating : TensorType,
    State <: BuilderState](optimizer: Optimizer[X, R])(implicit c: Convertible[Int, R]) {

    def using(alg: Algorithm): Builder[X, R, State with WithAlg] =
      copy(optimizer = optimizer.copy(alg = alg))

    def initWith(args: Shape => Tensor[R]): Builder[X, R, State] =
      copy(optimizer = optimizer.copy(initArgs = args))

    def on(dataset: RDD[Array[X]]): Builder[X, R, State with WithDataset] =
      copy(optimizer = optimizer.copy(dataset = dataset))

    def stopWhen(condition: Condition[R]): Builder[X, R, State with WithStopCondition] =
      copy(optimizer = optimizer.copy(stop = condition))

    def stopAfter(condition: Condition[R]): Builder[X, R, State with WithStopCondition] =
      stopWhen(condition)

    def epochs(number: Int): Builder[X, R, State with WithStopCondition] =
      stopWhen(Condition.epochs(number))

    def iterations(number: Int): Builder[X, R, State with WithStopCondition] =
      stopWhen(Condition.iterations(number))

    def partition(number: Int): Builder[X, R, State] =
      copy(optimizer = optimizer.copy(partitons = number))

    def batch(size: Int): Builder[X, R, State] =
      copy(optimizer = optimizer.copy(batchSize = size))

    // NOTE: figure out how to rename it to each without getting issues
    // when type inference is required
    def doEach(when: Condition[R], action: Step[R] => Unit): Builder[X, R, State] =
      each(when, Effect.stateless[Step[R]](action))

    def doEach(action: Step[R] => Unit): Builder[X, R, State] =
      each(always, Effect.stateless[Step[R]](action))

    def each(effect: Effect[_, Step[R]]): Builder[X, R, State] =
      each(always, effect)

    def each(when: Condition[R], effect: Effect[_, Step[R]]): Builder[X, R, State] = {
      copy(optimizer = optimizer.copy(doOnEach = optimizer.doOnEach :+ effect.conditional(when)))
    }

    def build(implicit ev: State =:= Complete): Optimizer[X, R] = optimizer
  }

  def minimize[E: Numeric: TensorType, R: Numeric: Floating : TensorType: Dist]
  (model: Model[E, R])(implicit c: Convertible[Int, R]): Builder[E, R, WithFunc] =
    Builder(Optimizer(null, model, s => Tensor.rand(s), null, 1, 10000, minimizing = true, always, Effects.empty))

  def maximize[E: Numeric: TensorType, R: Numeric: Floating : TensorType: Dist]
  (model: Model[E, R])(implicit c: Convertible[Int, R]): Builder[E, R, WithFunc] =
    Builder(Optimizer(null, model, s => Tensor.rand(s), null, 1, 10000, minimizing = false, always, Effects.empty))
}
