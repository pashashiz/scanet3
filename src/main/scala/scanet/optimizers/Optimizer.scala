package scanet.optimizers

import com.github.benmanes.caffeine.cache.Caffeine
import org.apache.spark.sql.Dataset
import scanet.core.{Tensor, _}
import scanet.math.syntax._
import scanet.models._
import scanet.optimizers.Condition.always
import scanet.optimizers.Optimizer.BuilderState._
import scanet.optimizers.syntax._

import java.time.Duration
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import scala.annotation.{nowarn, tailrec}
import scala.collection._
import scala.collection.immutable.Seq

case class Step[A: Numeric](batch: Int, epoch: Int = 0, iter: Int = 0) {
  def nextIter: Step[A] = incIter(1)
  def incIter(number: Int): Step[A] = copy(iter = iter + number)
  def nextEpoch: Step[A] = copy(epoch = epoch + 1)
  override def toString: String = s"$epoch:$iter"
}

case class StepResult[A: Numeric](
    iterations: Int,
    paramsDef: Params[ParamDef],
    params: Params[Tensor[A]],
    meta: Params[Tensor[A]],
    loss: A) {
  def paramsWithDef: Params[(ParamDef, Tensor[A])] =
    paramsDef.join(params)
}

case class StepContext[A: Numeric](
    step: Step[A],
    result: StepResult[A],
    lossModel: LossModel,
    time: Long)

// E - type of input dataset to train on, could have any numeric values
// R - type to use on a model, could be only Float or Double
case class Optimizer[A: Floating](
    alg: Algorithm,
    model: Model,
    loss: Loss,
    initParams: () => Option[Params[Tensor[A]]],
    dataset: Dataset[Record[A]],
    batchSize: Int,
    minimizing: Boolean,
    stop: Condition[StepContext[A]],
    boardDir: String = "board",
    @transient doOnEach: Seq[Effect[A]])(implicit c: Convertible[Int, A]) {

  private val lossModel = model.withLoss(loss)

  def run(): TrainedModel[A] = {
    val jobId = UUID.randomUUID().toString
    val ds: Dataset[Record[A]] = dataset.cache()
    val sc = ds.sparkSession.sparkContext
    val board = TensorBoard(boardDir)
    val shapes = ds.shapes
    println(s"Training model:\n${model.describe[A](batchSize +: shapes._1)}")

    @tailrec
    def optimize(
        prevStep: Step[A],
        params: Params[Tensor[A]],
        meta: Params[Tensor[A]]): Params[Tensor[A]] = {
      val weightsBr = sc.broadcast(params)
      val metaBr = sc.broadcast(meta)
      val start = System.currentTimeMillis()
      val result = ds.rdd
        .mapPartitions { it =>
          Iterator(
            optimizeOnPartition(jobId, it, shapes, prevStep.iter, weightsBr.value, metaBr.value))
        }
        .treeReduce(aggParamsAndMeta(jobId))
      val finish = System.currentTimeMillis()
      val step: Step[A] = prevStep.nextEpoch.incIter(result.iterations)
      val stepCtx = StepContext(step, result, lossModel, finish - start)

      doOnEach
        .foldLeft(Effect.State(Effect.Console(), board))((ctx, effect) => effect(ctx, stepCtx))
        .run()

      if (stop(stepCtx)) {
        result.params
      } else {
        optimize(step, result.params, result.meta)
      }
    }
    val params = optimize(Step(batchSize), Params.empty, Params.empty)

    lossModel.trained(params)
  }

  private def optimizeOnPartition(
      jobId: String,
      it: scala.Iterator[Record[A]],
      shapes: (Shape, Shape),
      globalIter: Int,
      params: Params[Tensor[A]],
      meta: Params[Tensor[A]]): StepResult[A] = {
    val resource = Optimizer.resource(jobId)
    val result = resource.sessionPool.within(session => {
      val batches = TensorIterator(it, shapes, batchSize)
      val batchedInputShape = batchSize +: shapes._1
      val paramsDef = model.params(batchedInputShape)
      val weightNames = paramsDef.filterValues(_.trainable).paths
      val (paramsInitialized, metaInitialized) =
        if (globalIter == 0) {
          val params = initParams()
            .getOrElse(paramsDef.mapValues(param => param.initialize[A].eval))
          val meta = paramsDef.filterValues(_.trainable)
            .mapValues(param => alg.initMeta[A](param.shape))
          (params, meta)
        } else {
          (params, meta)
        }
      val backprop = compileBackprop(resource.tfCache, session)
      val loss: (Tensor[A], Tensor[A], Params[Tensor[A]]) => (Tensor[A], Params[Tensor[A]]) =
        compileLoss(resource.tfCache, session)
      @tailrec
      def optimize(
          iter: Int,
          params: Params[Tensor[A]],
          meta: Params[Tensor[A]]): StepResult[A] = {
        val (x, y) = batches.next()
        val (weights, state) = params.partitionPaths(weightNames.contains)
        val (nextWeights, nextMeta, nextState) =
          backprop(x, y, weights, meta, state, Tensor.scalar(globalIter + iter + 1))
        val nextParams = nextWeights ++ nextState
        if (batches.hasNext) {
          optimize(iter + 1, nextParams, nextMeta)
        } else {
          val (lossResult, _) = loss(x, y, nextParams)
          StepResult(iter + 1, paramsDef, nextWeights, nextMeta, lossResult.toScalar)
        }
      }
      optimize(0, paramsInitialized, metaInitialized)
    })
    result
  }

  private def compileLoss(cache: Optimizer.Cache, session: Session) = {
    cache.getOrCompute(
      s"$lossModel:loss[${TensorType[A].classTag}]",
      lossModel.lossStateful[A].tf) compileWith session
  }

  private def compileBackprop(cache: Optimizer.Cache, session: Session) = {
    def newOutputSeq = Params[Expr[A]]()
    val tf = cache.getOrCompute(
      s"$lossModel:$alg:calc[${TensorType[A].classTag}]]", {
        val func = (
            x: Expr[A],
            y: Expr[A],
            weights: Params[Expr[A]],
            metas: Params[Expr[A]],
            state: Params[Expr[A]],
            iter: Expr[Int]) => {
          val (grads, nextState) = lossModel.gradStateful[A].apply(x, y, weights, state)
          val (nextAcc, nextMeta) = weights.join(grads).join(metas).params
            .foldLeft((newOutputSeq, newOutputSeq)) { (acc, next) =>
              val (gAcc, metaAcc) = acc
              val (path, ((w, g), meta)) = next
              val Delta(del, metaNext) = alg.delta[A](g, meta, iter)
              val d = del.cast[A]
              val gNext = if (minimizing) w - d else w + d
              (gAcc + (path -> gNext), metaAcc + (path -> metaNext))
            }
          (nextAcc, nextMeta, nextState)
        }
        func.tf
      })
    tf compileWith session
  }

  private def aggParamsAndMeta(jobId: String)(
      left: StepResult[A],
      right: StepResult[A]): StepResult[A] = {
    val resource = Optimizer.resource(jobId)
    resource.sessionPool.within { session =>
      def buildParamsAgg(
          leftParams: Params[Expr[A]],
          rightParams: Params[Expr[A]]): Params[Expr[A]] = {
        left.paramsDef.join(leftParams.join(rightParams)).mapValues {
          case (paramDef, (l, r)) =>
            paramDef.aggregation match {
              case Some(agg) => agg.build(Seq(l, r))
              case None      => paramDef.initializer.build[A](l.shape)
            }
        }
      }
      def buildMetaAgg(leftMeta: Params[Expr[A]], rightMeta: Params[Expr[A]]) = {
        (leftMeta join rightMeta).mapValues { case (l, r) => (l + r) / 2.0f.const.cast[A] }
      }
      val weightsAvg =
        resource.tfCache.getOrCompute("paramsAgg", (buildParamsAgg _).tf) compileWith session
      val metaAvg =
        resource.tfCache.getOrCompute("metaAvg", (buildMetaAgg _).tf) compileWith session
      val lossAvg = (left.loss plus right.loss) / c.convert(2)
      StepResult(
        left.iterations + right.iterations,
        left.paramsDef,
        weightsAvg(left.params, right.params),
        metaAvg(left.meta, right.meta),
        lossAvg)
    }
  }

}

object Optimizer {

  case class Resource(sessionPool: SessionPool, tfCache: Cache) extends AutoCloseable {
    override def close(): Unit = sessionPool.close()
  }

  class Cache {
    private val map = new ConcurrentHashMap[String, Any]()
    def getOrCompute[A](key: String, op: => A): A =
      map.computeIfAbsent(key, _ => op).asInstanceOf[A]
  }

  private def CPUs: Int = Runtime.getRuntime.availableProcessors()

  private val resources = Caffeine.newBuilder()
    .maximumSize(5)
    .expireAfterAccess(Duration.ofSeconds(300))
    .removalListener((_: String, resource: Resource, _) => resource.close())
    .build((_: String) => Resource(new SessionPool(CPUs), new Cache))

  def resource(id: String): Resource = resources.get(id)

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

    def initParams(args: => Params[Tensor[A]]): Builder[A, State] =
      copy(optimizer = optimizer.copy(initParams = () => Some(args)))

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

    def board(dir: String): Builder[A, State] =
      copy(optimizer = optimizer.copy(boardDir = dir))

    def eachIter(effect: Effect[A]): Builder[A, State] =
      each(always, effect)

    def each(when: Condition[StepContext[A]], effect: Effect[A]): Builder[A, State] = {
      copy(optimizer = optimizer.copy(doOnEach = optimizer.doOnEach :+ effect.conditional(when)))
    }

    @nowarn def build(implicit ev: State =:= Complete): Optimizer[A] = optimizer

    def run()(implicit ev: State =:= Complete): TrainedModel[A] = build.run()
  }

  def minimize[R: Floating](model: Model)(
      implicit c: Convertible[Int, R]): Builder[R, WithFunc] =
    Builder(
      Optimizer(
        alg = null,
        model = model,
        loss = null,
        initParams = () => None,
        dataset = null,
        batchSize = 10000,
        minimizing = true,
        stop = always,
        doOnEach = Seq(Effect.RecordIteration())))

  def maximize[R: Floating](model: Model)(
      implicit c: Convertible[Int, R]): Builder[R, WithFunc] =
    Builder(
      Optimizer(
        alg = null,
        model = model,
        loss = null,
        initParams = () => None,
        dataset = null,
        batchSize = 10000,
        minimizing = false,
        stop = always,
        doOnEach = Seq(Effect.RecordIteration())))
}
