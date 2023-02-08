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
      case out :: in :: receptive =>
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

  case class RandomUniform(min: Float = -0.05f, max: Float = 0.05f, seed: Option[Long] = None)
      extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] =
      rand[E](shape, Dist.Uniform, seed, scale = Some(max - min), shift = Some(min))
    override def toString: String = s"RandomUniform(min=$min,max=$max)"
  }

  case class RandomNormal(mean: Float = 0.0f, std: Float = 0.05f, seed: Option[Long] = None)
      extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] =
      rand[E](shape, Dist.Normal, seed, scale = Some(std), shift = Some(mean))
    override def toString: String = s"RandomNormal(mean=$mean,std=$std)"
  }

  case class RandomNormalTruncated(
      mean: Float = 0.0f,
      std: Float = 0.05f,
      seed: Option[Long] = None)
      extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] =
      rand[E](shape, Dist.NormalTruncated, seed, scale = Some(std), shift = Some(mean))
    override def toString: String = s"RandomNormalTruncated(mean=$mean,std=$std)"
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
      rand[E](shape, Dist.Uniform, seed, scale = Some(2 * limit), shift = Some(-limit))
    }
    override def toString: String = "GlorotUniform"
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
    override def toString: String = "GlorotNormal"
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
      rand[E](shape, Dist.Uniform, seed, scale = Some(2 * limit), shift = Some(-limit))
    }
    override def toString: String = "HeUniform"
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
    override def toString: String = "HeNormal"
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
      rand[E](shape, Dist.Uniform, seed, scale = Some(2 * limit), shift = Some(-limit))
    }
    override def toString: String = "LecunUniform"
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
    override def toString: String = "LecunNormal"
  }

  /** If the shape of the tensor to initialize is two-dimensional,
    * it is initialized with an orthogonal matrix obtained from the QR decomposition of a matrix
    * of random numbers drawn from a normal distribution.
    * If the matrix has fewer rows than columns then the output will have orthogonal rows.
    * Otherwise, the output will have orthogonal columns.
    *
    * If the shape of the tensor to initialize is more than two-dimensional,
    * a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])` is initialized,
    * where `n` is the length of the shape vector.
    * The matrix is subsequently reshaped to give a tensor of the desired shape.
    *
    * @param gain multiplicative factor to apply to the orthogonal matrix
    * @param seed seed for random generator
    */
  case class Orthogonal(gain: Option[Float] = None, seed: Option[Long] = None) extends Initializer {
    override def build[E: Floating](shape: Shape): Expr[E] = {
      require(shape.rank >= 2, s"at least rank 2 is required but was ${shape.rank}")
      val rows = shape.dropRight(1).power
      val cols = shape.last
      // we need rows to be always > cols, so we will get square R matrix
      val flatShape = Shape(math.max(rows, cols), math.min(rows, cols))
      val init = rand[E](flatShape, Dist.Normal, seed)
      // compute the QR factorization
      val (q, r) = init.qr()
      // make Q uniform
      val qu = q * r.diagPart.sign
      val qt = if (rows < cols) qu.transpose else qu
      val qg = gain.fold(qt)(g => qt * g.const.cast[E])
      qg.reshape(shape)
    }
  }
}
