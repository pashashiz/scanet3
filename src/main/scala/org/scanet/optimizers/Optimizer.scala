package org.scanet.optimizers

import org.apache.spark.rdd.RDD
import org.scanet.core.{Tensor, _}
import org.scanet.math.syntax._
import org.scanet.math.{Convertible, Dist, Numeric}
import org.scanet.models.Model
import org.scanet.optimizers.Condition.always
import org.scanet.optimizers.Optimizer.BuilderState._
import org.scanet.optimizers.Optimizer.{sessionsPool, tfCache}

import scala.annotation.tailrec
import scala.collection.mutable

case class Step[W: Numeric: TensorType, R: Numeric: TensorType](
    epoch: Int = 0, iter: Int = 0, result: Option[R] = None) {
  def nextIter: Step[W, R] = incIter(1)
  def incIter(number: Int): Step[W, R] = copy(iter = iter + number)
  def nextEpoch: Step[W, R] = copy(epoch = epoch + 1)
  def withResult(value: R): Step[W, R] = copy(result = Some(value))
  override def toString: String = s"$epoch:$iter"
}

case class Optimizer[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType]
     (alg: Algorithm,
      model: Model[X, W, R],
      initArgs: Shape => Tensor[W],
      dataset: RDD[Array[X]],
      partitons: Int,
      batchSize: Int,
      minimizing: Boolean,
      stop: Condition[W, R],
      @transient doOnEach: Effects[Step[W, R]])
     (implicit c: Convertible[Int, R]) {

  def run(): Tensor[W] = {
    val ds = dataset.repartition(partitons).cache()
    val sc = ds.sparkContext

    @tailrec
    def optimize(prevStep: Step[W, R], effectState: Seq[_], weights: Tensor[W], meta: Tensor[Float]): Tensor[W] = {
      val weightsBr = sc.broadcast(weights)
      val metaBr = sc.broadcast(meta)
      val (iter, newWeights, newMeta, result) = ds
        .mapPartitions(it => Iterator(optimizeOnPartition(
          it, prevStep.iter, weightsBr.value, metaBr.value)))
        .treeReduce(averageMetaAndWeights)
      val step: Step[W, R] = prevStep.nextEpoch.incIter(iter).withResult(result)
      val nextEffectState = doOnEach.action(effectState, step)
      if (stop(step)) {
        newWeights
      } else {
        optimize(step, nextEffectState, newWeights, newMeta)
      }
    }
    optimize(Step(), doOnEach.unit, Tensor.zeros(), Tensor.zeros())
  }

  private def optimizeOnPartition(
        it: scala.Iterator[Array[X]], globalIter: Int, weights: Tensor[W], meta: Tensor[Float]): (Int, Tensor[W], Tensor[Float], R) = {
    val result = sessionsPool.withing(session => {
      val batches: BatchingIterator[X] = BatchingIterator(it, batchSize)
      val (weightsInitialized, metaInitialized) = if (globalIter == 0) {
        val shape = model.shape(batches.columns)
        (initArgs(shape), alg.initMeta(shape))
      } else {
        (weights, meta)
      }

      val loss = tfCache.getOrCompute(s"$model:loss", model.loss) compile session
      val calc = tfCache.getOrCompute(s"$model:$alg:calc",
        TF2.identity[Float, Int].compose(model.weightsAndGrad) {
          case ((meta, iter), (w, g)) =>
            val Delta(del, nextMeta) = alg.delta(g, meta, iter)
            val d = del.cast[W]
            (if (minimizing) w - d else w + d, nextMeta)
        }.into[(Tensor[W], Tensor[Float])]) compile session

      @tailrec
      def optimize(iter: Int, weights: Tensor[W], meta: Tensor[Float]): (Int, Tensor[W], Tensor[Float], R) = {
        val batch = batches.next()
        val (nextWeights, nextMeta) = calc(meta, Tensor.scalar(globalIter + iter + 1), batch, weights)
        if (batches.hasNext) {
          optimize(iter + 1, nextWeights, nextMeta)
        } else {
          (iter + 1, nextWeights, nextMeta, loss(batch, nextWeights).toScalar)
        }
      }

      optimize(0, weightsInitialized, metaInitialized)
    })
    result
  }

  // todo: tuple3 -> case class
  private def averageMetaAndWeights(
       left: (Int, Tensor[W], Tensor[Float], R),
       right: (Int, Tensor[W], Tensor[Float], R)): (Int, Tensor[W], Tensor[Float], R) = {
    sessionsPool.withing(session => {
      val (leftIter, leftWeights, leftMeta, leftResult) = left
      val (rightIter, rightWeights, rightMeta, rightResult) = right
      val weightsAvg = tfCache.getOrCompute("weightsAvg", avg[W]) compile session
      val metaAvg = tfCache.getOrCompute("metaAvg", avg[Float]) compile session
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

  case class Builder[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType, State <: BuilderState](optimizer: Optimizer[X, W, R])(implicit c: Convertible[Int, R]) {

    def using(alg: Algorithm): Builder[X, W, R, State with WithAlg] =
      copy(optimizer = optimizer.copy(alg = alg))

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

    def partition(number: Int): Builder[X, W, R, State] =
      copy(optimizer = optimizer.copy(partitons = number))

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

  def minimize[X: Numeric: TensorType, W: Numeric: TensorType: Dist, R: Numeric: TensorType]
    (model: Model[X, W, R])(implicit c: Convertible[Int, R]): Builder[X, W, R, WithFunc] =
    Builder(Optimizer(null, model, s => Tensor.rand(s), null, 1, 10000, minimizing = true, always, Effects.empty))

  def maximize[X: Numeric: TensorType, W: Numeric: TensorType: Dist, R: Numeric: TensorType]
    (model: Model[X, W, R])(implicit c: Convertible[Int, R]): Builder[X, W, R, WithFunc] =
    Builder(Optimizer(null, model, s => Tensor.rand(s), null, 1, 10000, minimizing = false, always, Effects.empty))
}
