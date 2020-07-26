package org.scanet.optimizers

import scala.collection._
import org.apache.spark.rdd.RDD
import org.scanet.core.{Tensor, _}
import org.scanet.math.syntax._
import org.scanet.math.{Convertible, Dist, Floating, Numeric}
import org.scanet.models.{Loss, Model, TrainedModel}
import org.scanet.optimizers.Condition.always
import org.scanet.optimizers.Optimizer.BuilderState._
import org.scanet.optimizers.Optimizer.{sessionsPool, tfCache}

import scala.annotation.tailrec
import scala.collection.mutable

case class Step[A: Numeric: TensorType](
    epoch: Int = 0, iter: Int = 0, result: Option[A] = None) {
  def nextIter: Step[A] = incIter(1)
  def incIter(number: Int): Step[A] = copy(iter = iter + number)
  def nextEpoch: Step[A] = copy(epoch = epoch + 1)
  def withResult(value: A): Step[A] = copy(result = Some(value))
  override def toString: String = s"$epoch:$iter"
}

// E - type of input dataset to train on, could have any numeric values
// R - type to use on a model, could be only Float or Double
case class Optimizer[
  A: Numeric : Floating : TensorType](
     alg: Algorithm,
     model: Model,
     loss: Loss,
     initArgs: Shape => Tensor[A],
     dataset: RDD[Array[A]],
     partitons: Int,
     batchSize: Int,
     minimizing: Boolean,
     stop: Condition[A],
     @transient doOnEach: Effects[Step[A]])
   (implicit c: Convertible[Int, A]) {

  case class IterResult(iter: Int, weights: Seq[Tensor[A]], meta: Seq[Tensor[A]], loss: A)

  private val lossModel = model.withLoss(loss)

  def run(): TrainedModel[A] = {
    val ds = dataset.repartition(partitons).cache()
    val sc = ds.sparkContext

    @tailrec
    def optimize(prevStep: Step[A], effectState: Seq[_], weights: Seq[Tensor[A]], meta: Seq[Tensor[A]]): Seq[Tensor[A]] = {
      val weightsBr = sc.broadcast(weights)
      val metaBr = sc.broadcast(meta)
      val result = ds
        .mapPartitions(it => Iterator(optimizeOnPartition(
          it, prevStep.iter, weightsBr.value, metaBr.value)))
        .treeReduce(averageMetaAndWeights)
      val step: Step[A] = prevStep.nextEpoch.incIter(result.iter).withResult(result.loss)
      val nextEffectState = doOnEach.action(effectState, step)
      if (stop(step)) {
        result.weights
      } else {
        optimize(step, nextEffectState, result.weights, result.meta)
      }
    }
    val weights = optimize(Step(), doOnEach.unit, Seq(), Seq())
    lossModel.trained(weights)
  }

  private def optimizeOnPartition(
    it: scala.Iterator[Array[A]], globalIter: Int, weights: Seq[Tensor[A]], meta: Seq[Tensor[A]]): IterResult = {
    val result = sessionsPool.withing(session => {
      val batches = Tensor2Iterator(it, batchSize, splitAt = size => size - model.outputs())
      val (weightsInitialized, metaInitialized) = if (globalIter == 0) {
        val features = batches.columns - model.outputs()
        val shapes = model.shapes(features)
        (shapes.map(initArgs(_)), shapes.map(alg.initMeta[A](_)))
      } else {
        (weights, meta)
      }
      val loss = compileLoss(session)
      val calc = compileCalc(session)
      @tailrec
      def optimize(iter: Int, weights: Seq[Tensor[A]], meta: Seq[Tensor[A]]): IterResult = {
        val (x, y) = batches.next()
        /*_*/
        val (nextWeights, nextMeta) = calc(x, y, weights, meta, Tensor.scalar(globalIter + iter + 1))
        /*_*/
        if (batches.hasNext) {
          optimize(iter + 1, nextWeights, nextMeta)
        } else {
          IterResult(iter + 1, nextWeights, nextMeta, loss(x, y, nextWeights).toScalar)
        }
      }
      optimize(0, weightsInitialized, metaInitialized)
    })
    result
  }

  private def compileLoss(session: Session) = {
    tfCache.getOrCompute(
      s"$lossModel:loss[${TensorType[A].classTag}]",
      lossModel.loss[A] compile session)
  }

  private def compileCalc(session: Session) = {
    def newOutputSeq: OutputSeq[A] = Seq[Output[A]]()
    tfCache.getOrCompute(
      s"$lossModel:$alg:calc[${TensorType[A].classTag}]]",
      lossModel.weightsAndGrad[A].combine(TF2.identity[OutputSeq, A, Output, Int]) {
        case ((ws, gs), (metas, iter)) =>
          (ws, gs, metas).zipped.foldLeft((newOutputSeq, newOutputSeq))((acc, next) => {
            val (gAcc, metaAcc) = acc
            val (w, g, meta) = next
            val Delta(del, metaNext) = alg.delta[A](g, meta, iter)
            val d = del.cast[A]
            val gNext = if (minimizing) w - d else w + d
            (gAcc :+ gNext, metaAcc :+ metaNext)
          })
      } compile session)
  }

  private def averageMetaAndWeights(left: IterResult, right: IterResult): IterResult = {
    sessionsPool.withing(session => {
      val weightsAvg = tfCache.getOrCompute("weightsAvg", avg[A]) compile session
      val metaAvg = tfCache.getOrCompute("metaAvg", avg[A]) compile session
      val lossAvg = (left.loss plus right.loss) / c.convert(2)
      IterResult(
        left.iter + right.iter,
        weightsAvg(left.weights, right.weights),
        metaAvg(left.meta, right.meta),
        lossAvg)
    })
  }

  private def avg[X: Numeric: TensorType]: TF2[X, Seq[Tensor[X]], X, Seq[Tensor[X]], OutputSeq[X]] =
    TF2[OutputSeq, X, OutputSeq, X, OutputSeq[X]]((arg1, arg2) => {
      (arg1 zip arg2).map { case (l, r) => (l + r) / 2.0f.const.cast[X] }
    })
}

object Optimizer {

  class Cache {
    private val map = concurrent.TrieMap[String, Any]()
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
    sealed trait WithLoss extends BuilderState
    sealed trait WithDataset extends BuilderState
    sealed trait WithStopCondition extends BuilderState
    type Complete = WithAlg with WithFunc with WithLoss with WithDataset with WithStopCondition
  }

  case class Builder[A: Numeric: Floating : TensorType, State <: BuilderState]
  (optimizer: Optimizer[A])(implicit c: Convertible[Int, A]) {

    def loss(loss: Loss): Builder[A, State with WithLoss] =
      copy(optimizer = optimizer.copy(loss = loss))

    def using(alg: Algorithm): Builder[A, State with WithAlg] =
      copy(optimizer = optimizer.copy(alg = alg))

    def initWith(args: Shape => Tensor[A]): Builder[A, State] =
      copy(optimizer = optimizer.copy(initArgs = args))

    def on(dataset: RDD[Array[A]]): Builder[A, State with WithDataset] =
      copy(optimizer = optimizer.copy(dataset = dataset))

    def stopWhen(condition: Condition[A]): Builder[A, State with WithStopCondition] =
      copy(optimizer = optimizer.copy(stop = condition))

    def stopAfter(condition: Condition[A]): Builder[A, State with WithStopCondition] =
      stopWhen(condition)

    def epochs(number: Int): Builder[A, State with WithStopCondition] =
      stopWhen(Condition.epochs(number))

    def iterations(number: Int): Builder[A, State with WithStopCondition] =
      stopWhen(Condition.iterations(number))

    def partition(number: Int): Builder[A, State] =
      copy(optimizer = optimizer.copy(partitons = number))

    def batch(size: Int): Builder[A, State] =
      copy(optimizer = optimizer.copy(batchSize = size))

    // NOTE: figure out how to rename it to each without getting issues
    // when type inference is required
    def doEach(when: Condition[A], action: Step[A] => Unit): Builder[A, State] =
      each(when, Effect.stateless[Step[A]](action))

    def doEach(action: Step[A] => Unit): Builder[A, State] =
      each(always, Effect.stateless[Step[A]](action))

    def each(effect: Effect[_, Step[A]]): Builder[A, State] =
      each(always, effect)

    def each(when: Condition[A], effect: Effect[_, Step[A]]): Builder[A, State] = {
      copy(optimizer = optimizer.copy(doOnEach = optimizer.doOnEach :+ effect.conditional(when)))
    }

    def build(implicit ev: State =:= Complete): Optimizer[A] = optimizer

    def run()(implicit ev: State =:= Complete): TrainedModel[A] = build.run()
  }

  def minimize[R: Numeric: Floating : TensorType: Dist]
  (model: Model)(implicit c: Convertible[Int, R]): Builder[R, WithFunc] =
    Builder(Optimizer(null, model, null, s => Tensor.rand(s), null, 1, 10000, minimizing = true, always, Effects.empty))

  def maximize[R: Numeric: Floating : TensorType: Dist]
  (model: Model)(implicit c: Convertible[Int, R]): Builder[R, WithFunc] =
    Builder(Optimizer(null, model, null, s => Tensor.rand(s), null, 1, 10000, minimizing = false, always, Effects.empty))
}
