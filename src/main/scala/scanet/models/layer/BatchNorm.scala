package scanet.models.layer

import scanet.core._
import scanet.math.syntax.zeros
import scanet.models.Aggregation.Avg
import scanet.models.layer.BatchNorm.{Beta, Gamma, MovingMean, MovingVariance}
import scanet.models.{Initializer, ParamDef, Regularization}
import scanet.syntax._

/** Layer that normalizes its inputs.
  *
  * Batch normalization applies a transformation that maintains the mean output
  * close to 0 and the output standard deviation close to 1.
  *
  * Importantly, batch normalization works differently during training and
  * during inference.
  *
  * '''During training''', the layer normalizes its output using
  * the mean and standard deviation of the current batch of inputs. That is to
  * say, for each channel being normalized, the layer returns
  *
  * {{{gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta}}}
  *
  * where:
  *   - `epsilon` is small constant (configurable as part of the constructor arguments)
  *   - `gamma` is a learned scaling factor (initialized as 1)
  *   - `beta` is a learned offset factor (initialized as 0)
  *
  * '''During inference''' the layer normalizes its output using a moving average of the
  * mean and standard deviation of the batches it has seen during training. That
  * is to say, it returns
  *
  * {{{gamma * (batch - moving_mean) / sqrt(moving_var + epsilon) + beta}}}
  *
  * where `moving_mean` and `moving_var` are non-trainable variables that
  * are updated each time the layer in called in training mode, as such:
  *   - `moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)`
  *   - `moving_var = moving_var * momentum + var(batch) * (1 - momentum)`
  *
  * As such, the layer will only normalize its inputs during inference
  * after having been trained on data that has similar statistics as the inference data.
  *
  * Reference [Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167).
  *
  * @param axis Axis that should be normalized (typically the features axis).
  * @param momentum Momentum for the moving average.
  * @param epsilon Small float added to variance to avoid dividing by zero.
  * @param betaInitializer Initializer for the beta weight
  * @param gammaInitializer Initializer for the gamma weight.
  * @param movingMeanInitializer Initializer for the moving mean.
  * @param movingVarianceInitializer Initializer for the moving variance.
  * @param betaRegularizer Regularizer for the beta weight.
  * @param gammaRegularizer Regularizer for the gamma weight.
  * @param trainable Whether layer is trainable
  */
case class BatchNorm(
    axis: Seq[Int] = Seq(-1),
    momentum: Float = 0.99f,
    epsilon: Float = 1e-3f,
    betaInitializer: Initializer = Initializer.Zeros,
    gammaInitializer: Initializer = Initializer.Ones,
    movingMeanInitializer: Initializer = Initializer.Zeros,
    movingVarianceInitializer: Initializer = Initializer.Ones,
    betaRegularizer: Regularization = Regularization.Zero,
    gammaRegularizer: Regularization = Regularization.Zero,
    override val trainable: Boolean = true)
    extends Layer {

  override def stateful: Boolean = true

  private def paramsShape(input: Shape): Shape = {
    // given shape = (2, 4, 6) and axis = (1, 2)
    // we will end up with specified axis keeping their dimension, while the rest reduced to 1
    // so result shape = (1, 4, 6)
    // note 1: in case of reduction operation, such as mean - it works vice-versa, specified dimension will become 1
    // note 2: we keep all 1 dimensions without squeezing to perform proper broadcast with complex deep shapes
    val reduceAxis = input.axisExcept(axis: _*)
    input.updateAll(1)(reduceAxis: _*)
  }

  override def params(input: Shape): Params[ParamDef] = {
    val shape = paramsShape(input)
    Params(
      Beta -> ParamDef(shape, betaInitializer, Some(Avg), trainable = trainable),
      Gamma -> ParamDef(shape, gammaInitializer, Some(Avg), trainable = trainable),
      MovingMean -> ParamDef(shape, movingMeanInitializer, Some(Avg)),
      MovingVariance -> ParamDef(shape, movingVarianceInitializer, Some(Avg)))
  }

  override def build[E: Floating](
      input: Expr[E],
      params: Params[Expr[E]]): (Expr[E], Params[Expr[E]]) = {
    val prevMovingMean = params(MovingMean)
    val prevMovingVariance = params(MovingVariance)
    val (movingMean, movingVariance) =
      if (trainable) {
        val momentumE = momentum.const.cast[E]
        val reduceAxis = input.shape.axisExcept(axis: _*)
        val batchMean = input.mean(reduceAxis, keepDims = true)
        val batchVariance = (input - batchMean).sqr.mean(reduceAxis, keepDims = true)
        val movingMean = prevMovingMean.decayingAvg(batchMean, momentumE)
        val movingVariance = prevMovingVariance.decayingAvg(batchVariance, momentumE)
        (movingMean, movingVariance)
      } else {
        (prevMovingMean, prevMovingVariance)
      }
    val epsilonE = epsilon.const.cast[E]
    val output =
      (input - movingMean) * params(Gamma) /
      (movingVariance.sqrt + epsilonE) - params(Beta)
    val nextState: Params[Expr[E]] =
      if (trainable) Params(
        MovingMean -> movingMean,
        MovingVariance -> movingVariance)
      else Params.empty
    (output, nextState)
  }

  override def penalty[E: Floating](params: Params[Expr[E]]): Expr[E] =
    if (trainable) betaRegularizer.build(params(Gamma)) + betaRegularizer.build(params(Gamma))
    else zeros[E](Shape())

  override def outputShape(input: Shape): Shape = input

  override def makeTrainable(trainable: Boolean): BatchNorm = copy(trainable = trainable)

  override def toString: String = s"BatchNorm($axis)"
}

object BatchNorm {
  val Beta: Path = "beta"
  val Gamma: Path = "gamma"
  val MovingMean: Path = "moving_mean"
  val MovingVariance: Path = "moving_variance"
}
