package org.scanet.optimizers

import org.apache.spark.rdd.RDD
import org.scanet.core.{Tensor, _}
import org.scanet.math.syntax._
import org.scanet.math.{Convertible, Dist, Floating, Numeric}
import org.scanet.models.{Loss, LossModel, Model, TrainedModel}
import org.scanet.optimizers.Condition.always
import org.scanet.optimizers.Optimizer.BuilderState._
import org.scanet.optimizers.Optimizer.{sessionsPool, tfCache}

import scala.annotation.tailrec
import scala.collection._

case class Step[A: Numeric: TensorType](epoch: Int = 0, iter: Int = 0) {
  def nextIter: Step[A] = incIter(1)
  def incIter(number: Int): Step[A] = copy(iter = iter + number)
  def nextEpoch: Step[A] = copy(epoch = epoch + 1)
  override def toString: String = s"$epoch:$iter"
}

case class StepResult[A: Numeric: TensorType](iterations: Int, weights: Seq[Tensor[A]], meta: Seq[Tensor[A]], loss: A)

case class StepContext[A: Numeric: TensorType](step: Step[A], result: StepResult[A], lossModel: LossModel)

// E - type of input dataset to train on, could have any numeric values
// R - type to use on a model, could be only Float or Double
case class Optimizer[
  A: Numeric : Floating : TensorType](
     alg: Algorithm,
     model: Model,
     loss: Loss,
     initArgs: Shape => Tensor[A],
     dataset: RDD[Array[A]],
     batchSize: Int,
     minimizing: Boolean,
     stop: Condition[StepContext[A]],
     @transient doOnEach: Seq[Effect[StepContext[A]]])
   (implicit c: Convertible[Int, A]) {

  private val lossModel = model.withLoss(loss)

  def run(): TrainedModel[A] = {
    val ds = dataset.cache()
    val sc = ds.sparkContext

    @tailrec
    def optimize(prevStep: Step[A], weights: Seq[Tensor[A]], meta: Seq[Tensor[A]]): Seq[Tensor[A]] = {
      val weightsBr = sc.broadcast(weights)
      val metaBr = sc.broadcast(meta)
      val result = ds
        .mapPartitions(it => Iterator(optimizeOnPartition(
          it, prevStep.iter, weightsBr.value, metaBr.value)))
        .treeReduce(averageMetaAndWeights)
      val step: Step[A] = prevStep.nextEpoch.incIter(result.iterations)
      val stepCtx = StepContext(step, result, lossModel)
      doOnEach.foreach(effect => effect(stepCtx))
      if (stop(stepCtx)) {
        result.weights
      } else {
        optimize(step, result.weights, result.meta)
      }
    }
    val weights = optimize(Step(), Seq(), Seq())
    lossModel.trained(weights)
  }

  private def optimizeOnPartition(
    it: scala.Iterator[Array[A]], globalIter: Int, weights: Seq[Tensor[A]], meta: Seq[Tensor[A]]): StepResult[A] = {
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
      def optimize(iter: Int, weights: Seq[Tensor[A]], meta: Seq[Tensor[A]]): StepResult[A] = {
        val (x, y) = batches.next()
        /*_*/
        val (nextWeights, nextMeta) = calc(x, y, weights, meta, Tensor.scalar(globalIter + iter + 1))
        /*_*/
        if (batches.hasNext) {
          optimize(iter + 1, nextWeights, nextMeta)
        } else {
          StepResult(iter + 1, nextWeights, nextMeta, loss(x, y, nextWeights).toScalar)
        }
      }
      optimize(0, weightsInitialized, metaInitialized)
    })
    result
  }

  private def compileLoss(session: Session) = {
    tfCache.getOrCompute(s"$lossModel:loss[${TensorType[A].classTag}]", lossModel.loss[A]) compile session
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
      }) compile session
  }

  private def averageMetaAndWeights(left: StepResult[A], right: StepResult[A]): StepResult[A] = {
    sessionsPool.withing(session => {
      val weightsAvg = tfCache.getOrCompute("weightsAvg", avg[A]) compile session
      val metaAvg = tfCache.getOrCompute("metaAvg", avg[A]) compile session
      val lossAvg = (left.loss plus right.loss) / c.convert(2)
      StepResult(
        left.iterations + right.iterations,
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

    def stopWhen(condition: Condition[StepContext[A]]): Builder[A, State with WithStopCondition] =
      copy(optimizer = optimizer.copy(stop = condition))

    def stopAfter(condition: Condition[StepContext[A]]): Builder[A, State with WithStopCondition] =
      stopWhen(condition)

    def epochs(number: Int): Builder[A, State with WithStopCondition] =
      stopWhen(Condition.epochs(number))

    def iterations(number: Int): Builder[A, State with WithStopCondition] =
      stopWhen(Condition.iterations(number))

    def batch(size: Int): Builder[A, State] =
      copy(optimizer = optimizer.copy(batchSize = size))

    def eachIter(effect: Effect[StepContext[A]]): Builder[A, State] =
      each(always, effect)

    def each(when: Condition[StepContext[A]], effect: Effect[StepContext[A]]): Builder[A, State] = {
      copy(optimizer = optimizer.copy(doOnEach = optimizer.doOnEach :+ effect.conditional(when)))
    }

    def build(implicit ev: State =:= Complete): Optimizer[A] = optimizer

    def run()(implicit ev: State =:= Complete): TrainedModel[A] = build.run()
  }

  def minimize[R: Numeric: Floating : TensorType: Dist]
  (model: Model)(implicit c: Convertible[Int, R]): Builder[R, WithFunc] =
    Builder(Optimizer(null, model, null, s => Tensor.rand(s, range = Some((Numeric[R].one.negate, Numeric[R].one))), null, 10000, minimizing = true, always, Seq()))

  def maximize[R: Numeric: Floating : TensorType: Dist]
  (model: Model)(implicit c: Convertible[Int, R]): Builder[R, WithFunc] =
    Builder(Optimizer(null, model, null, s => Tensor.rand(s, range = Some((Numeric[R].one.negate, Numeric[R].one))), null, 10000, minimizing = false, always, Seq()))
}
