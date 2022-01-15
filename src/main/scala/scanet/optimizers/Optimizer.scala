package scanet.optimizers

import org.apache.spark.sql.Dataset
import scanet.core.{Tensor, _}
import scanet.math.syntax._
import scanet.math.Dist
import scanet.models.{Loss, LossModel, Model, TrainedModel}
import scanet.optimizers.syntax._
import scanet.optimizers.Condition.always
import scanet.optimizers.Optimizer.BuilderState._
import scanet.optimizers.Optimizer.{sessionsPool, tfCache}

import scala.annotation.tailrec
import scala.collection._
import scala.collection.immutable.Seq

case class Step[A: Numeric](epoch: Int = 0, iter: Int = 0) {
  def nextIter: Step[A] = incIter(1)
  def incIter(number: Int): Step[A] = copy(iter = iter + number)
  def nextEpoch: Step[A] = copy(epoch = epoch + 1)
  override def toString: String = s"$epoch:$iter"
}

case class StepResult[A: Numeric](
    iterations: Int,
    weights: Seq[Tensor[A]],
    meta: Seq[Tensor[A]],
    loss: A)

case class StepContext[A: Numeric](
    step: Step[A],
    result: StepResult[A],
    lossModel: LossModel)

// E - type of input dataset to train on, could have any numeric values
// R - type to use on a model, could be only Float or Double
case class Optimizer[A: Floating](
    alg: Algorithm,
    model: Model,
    loss: Loss,
    initArgs: Shape => Tensor[A],
    dataset: Dataset[Record[A]],
    batchSize: Int,
    minimizing: Boolean,
    stop: Condition[StepContext[A]],
    @transient doOnEach: Seq[Effect[StepContext[A]]])(implicit c: Convertible[Int, A]) {

  private val lossModel = model.withLoss(loss)

  def run(): TrainedModel[A] = {
    val ds: Dataset[Record[A]] = dataset.cache()
    val sc = ds.sparkSession.sparkContext

    @tailrec
    def optimize(
        prevStep: Step[A],
        weights: Seq[Tensor[A]],
        meta: Seq[Tensor[A]]): Seq[Tensor[A]] = {
      val weightsBr = sc.broadcast(weights)
      val metaBr = sc.broadcast(meta)
      val shapes = ds.shapes
      val result = ds.rdd
        .mapPartitions(it =>
          Iterator(optimizeOnPartition(
            it,
            shapes,
            prevStep.iter,
            weightsBr.value,
            metaBr.value)))
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
      it: scala.Iterator[Record[A]],
      shapes: (Shape, Shape),
      globalIter: Int,
      weights: Seq[Tensor[A]],
      meta: Seq[Tensor[A]]): StepResult[A] = {
    val result = sessionsPool.withing(session => {
      val batches = TensorIterator(it, shapes, batchSize)
      val (weightsInitialized, metaInitialized) =
        if (globalIter == 0) {
          // todo: run tests
          val weightShapes = model.weightsShapes(shapes._1)
          (weightShapes.map(initArgs(_)).toList, weightShapes.map(alg.initMeta[A](_)).toList)
        } else {
          (weights, meta)
        }
      val loss = compileLoss(session)
      val calc = compileCalc(session)
      @tailrec
      def optimize(iter: Int, weights: Seq[Tensor[A]], meta: Seq[Tensor[A]]): StepResult[A] = {
        val (x, y) = batches.next()
        /*_*/
        val (nextWeights, nextMeta) =
          calc(x, y, weights, meta, Tensor.scalar(globalIter + iter + 1))
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
    tfCache.getOrCompute(
      s"$lossModel:loss[${TensorType[A].classTag}]",
      lossModel.loss[A]) compile session
  }

  private def compileCalc(session: Session) = {
    def newOutputSeq: OutputSeq[A] = Seq[Expr[A]]()
    tfCache.getOrCompute(
      s"$lossModel:$alg:calc[${TensorType[A].classTag}]]",
      lossModel.weightsAndGrad[A].combine(TF2.identity[OutputSeq, A, Expr, Int]) {
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

  private def avg[X: Numeric]: TF2[X, Seq[Tensor[X]], X, Seq[Tensor[X]], OutputSeq[X]] =
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
        case None    => val d = op; map(key) = d; d
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

  case class Builder[A: Floating, State <: BuilderState](
      optimizer: Optimizer[A])(implicit c: Convertible[Int, A]) {

    def loss(loss: Loss): Builder[A, State with WithLoss] =
      copy(optimizer = optimizer.copy(loss = loss))

    def using(alg: Algorithm): Builder[A, State with WithAlg] =
      copy(optimizer = optimizer.copy(alg = alg))

    def initWith(args: Shape => Tensor[A]): Builder[A, State] =
      copy(optimizer = optimizer.copy(initArgs = args))

    def on(dataset: Dataset[Record[A]]): Builder[A, State with WithDataset] =
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

  def minimize[R: Floating: Dist](model: Model)(
      implicit c: Convertible[Int, R]): Builder[R, WithFunc] =
    Builder(
      Optimizer(
        alg = null,
        model = model,
        loss = null,
        initArgs = s => Tensor.rand(s, range = Some((Numeric[R].one.negate, Numeric[R].one))),
        dataset = null,
        batchSize = 10000,
        minimizing = true,
        stop = always,
        doOnEach = Seq()))

  def maximize[R: Floating: Dist](model: Model)(
      implicit c: Convertible[Int, R]): Builder[R, WithFunc] =
    Builder(
      Optimizer(
        alg = null,
        model = model,
        loss = null,
        initArgs = s => Tensor.rand(s, range = Some((Numeric[R].one.negate, Numeric[R].one))),
        dataset = null,
        batchSize = 10000,
        minimizing = false,
        stop = always,
        doOnEach = Seq()))
}
