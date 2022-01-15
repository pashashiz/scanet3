package scanet.models

import scanet.core.{Expr, Floating, Shape}
import scanet.math.rand.Dist
import scanet.math.syntax._

trait Initializer {

  def build[E: Floating](shape: Shape): Expr[E]

  /** Transforms a shape into fan-in (features in) and fan-out (features out),
    * where first dimension becomes in and last out
    *
    * If shape has rank > 2 we take a power of first dimensions and use it as multiplier for in and out
    */
  def fans(shape: Shape): (Int, Int) = {
    shape.dims match {
      case Nil              => (1, 1)
      case out :: Nil       => (out, out)
      case in :: out :: Nil => (in, out)
      case receptive :+ in :+ out =>
        val size = receptive.product
        (size * in, size * out)
    }
  }
}

object Initializer {

  private val TruncCoef: Float = 0.87962566103423978f

  case object Zeros extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] = zeros(shape)
    override def toString: String = "Zeros"
  }

  case object Ones extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] = ones(shape)
    override def toString: String = "Ones"
  }

  case class RandomUniform(min: Float = -0.05f, max: Float = 0.05f, seed: Option[Long])
      extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] =
      rand[E](shape, Dist.Uniform, seed, scale = Some((max - min) / 2), shift = Some(min + max))
  }

  case class RandomNormal(mean: Float = 0.0f, std: Float = 0.05f, seed: Option[Long])
      extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] =
      rand[E](shape, Dist.Normal, seed, scale = Some(std), shift = Some(mean))
  }

  case class RandomNormalTruncated(mean: Float = 0.0f, std: Float = 0.05f, seed: Option[Long])
      extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] =
      rand[E](shape, Dist.NormalTruncated, seed, scale = Some(std), shift = Some(mean))
  }

  /** The Glorot uniform initializer, also called Xavier uniform initializer.
    *
    * Draws samples from a uniform distribution within `[-limit, limit]`, where
    * {{{
    *   limit = sqrt(6 / (fan_in + fan_out))
    * }}}
    * (`fan_in` is the number of input units
    * in the weight tensor and `fan_out` is the number of output units).
    *
    * @param seed seed for random generator
    */
  case class GlorotUniform(seed: Option[Long] = None) extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] = {
      val (fanIn, fanOut) = fans(shape)
      val limit = math.sqrt(6f / (fanIn + fanOut)).toFloat
      rand[E](shape, Dist.Uniform, seed, scale = Some(limit))
    }
  }

  /** The Glorot normal initializer, also called Xavier normal initializer.
    *
    * Draws samples from a truncated normal distribution centered on 0 with
    * {{{
    *   std = sqrt(2 / (fan_in + fan_out))
    * }}}
    * where `fan_in` is the number of input units
    * in the weight tensor and `fan_out` is the number of output units in the weight tensor.
    *
    * @param seed seed for random generator
    */
  case class GlorotNormal(seed: Option[Long] = None) extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] = {
      val (fanIn, fanOut) = fans(shape)
      val std = math.sqrt(2f / (fanIn + fanOut)).toFloat / TruncCoef
      rand[E](shape, Dist.NormalTruncated, seed, scale = Some(std))
    }
  }

  /** He uniform variance scaling initializer.
    *
    * Draws samples from a uniform distribution within `[-limit, limit]`, where
    * {{{
    *   limit = sqrt(6 / fan_in)
    * }}}
    * (`fan_in` is the number of input units in the weight tensor).
    *
    * @param seed seed for random generator
    */
  case class HeUniform(seed: Option[Long] = None) extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] = {
      val (fanIn, _) = fans(shape)
      val limit = math.sqrt(6f / fanIn).toFloat
      rand[E](shape, Dist.Uniform, seed, scale = Some(limit))
    }
  }

  /** He normal initializer.
    *
    * It draws samples from a truncated normal distribution centered on 0 with
    * {{{
    *   stddev = sqrt(2 / fan_in)
    * }}}
    * where `fan_in` is the number of input units in the weight tensor.
    *
    * @param seed seed for random generator
    */
  case class HeNormal(seed: Option[Long] = None) extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] = {
      val (fanIn, _) = fans(shape)
      val std = math.sqrt(2f / fanIn).toFloat / TruncCoef
      rand[E](shape, Dist.NormalTruncated, seed, scale = Some(std))
    }
  }

  /** Lecun uniform initializer.
    *
    * Draws samples from a uniform distribution within `[-limit, limit]`, where
    * {{{
    *   limit = sqrt(3 / fan_in)
    * }}}
    * (`fan_in` is the number of input units in the weight tensor)..
    *
    * @param seed seed for random generator
    */
  case class LecunUniform(seed: Option[Long] = None) extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] = {
      val (fanIn, _) = fans(shape)
      val limit = math.sqrt(3f / fanIn).toFloat
      rand[E](shape, Dist.Uniform, seed, scale = Some(limit))
    }
  }

  /** Lecun normal initializer.
    *
    * Draws samples from a truncated normal distribution centered on 0 with
    * {{{
    *   stddev = sqrt(1 / fan_in)
    * }}}
    * where `fan_in` is the number of input units in the weight tensor.
    *
    * @param seed seed for random generator
    */
  case class LecunNormal(seed: Option[Long] = None) extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] = {
      val (fanIn, _) = fans(shape)
      val std = math.sqrt(1f / fanIn).toFloat / TruncCoef
      rand[E](shape, Dist.NormalTruncated, seed, scale = Some(std))
    }
  }
}
