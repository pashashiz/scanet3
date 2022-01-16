package scanet.math.rand

import scanet.core
import scanet.core.syntax._
import scanet.math.alg.kernels.syntax._
import scanet.core._
import scanet.math.alg.AllKernels
import scanet.math.alg.kernels.AllSyntax
import scanet.math.rand.Dist._

import scala.collection.immutable.Seq

case class RandomUniform[A: Floating](shape: Shape, seed: Option[Long]) extends Expr[A] {
  override def name: String = "RandomUniform"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def inputs: Seq[Expr[_]] = Seq(shape.toLongTensor.const)
  override def compiler: core.Compiler[A] = {
    val comp = DefaultCompiler[A]()
      .withAttr("dtype", TensorType[A])
    seed.map(comp.withAttr("seed", _)).getOrElse(comp)
  }
}

case class RandomNormal[A: Floating](shape: Shape, seed: Option[Long]) extends Expr[A] {
  override def name: String = "RandomStandardNormal"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def inputs: Seq[Expr[_]] = Seq(shape.toLongTensor.const)
  override def compiler: core.Compiler[A] = {
    val comp = DefaultCompiler[A]()
      .withAttr("dtype", TensorType[A])
    seed.map(comp.withAttr("seed", _)).getOrElse(comp)
  }
}

case class RandomNormalTruncated[A: Floating](shape: Shape, seed: Option[Long]) extends Expr[A] {
  override def name: String = "TruncatedNormal"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def inputs: Seq[Expr[_]] = Seq(shape.toLongTensor.const)
  override def compiler: core.Compiler[A] = {
    val comp = DefaultCompiler[A]()
      .withAttr("dtype", TensorType[A])
    seed.map(comp.withAttr("seed", _)).getOrElse(comp)
  }
}

sealed trait Dist
object Dist {
  case object Uniform extends Dist
  case object Normal extends Dist
  case object NormalTruncated extends Dist
}

trait AllKernels {

  /** Generate random tensor with a given shape.
    *
    * @param shape tensor shape
    * @param dist random distribution, could be:
    *             - [[Uniform]] Outputs random values from a uniform distribution.
    *               The generated values follow a uniform distribution in the range `[0, 1)`.
    *             - [[Normal]] Outputs random values from a normal distribution.
    *               The generated values will have mean 0 and standard deviation 1.
    *             - [[NormalTruncated]] Outputs random values from a truncated normal distribution.
    *               The generated values follow a normal distribution with mean 0 and standard deviation 1,
    *               except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
    * @param seed the seed value to start generation from, when [[None]] the random value will be used
    * @param scale the value to multiply the generated tensor by
    */
  def rand[A: Floating](
      shape: Shape,
      dist: Dist = Normal,
      seed: Option[Long] = None,
      scale: Option[Float] = None,
      shift: Option[Float] = None): Expr[A] = {
    val expr = dist match {
      case Uniform         => RandomUniform[A](shape, seed)
      case Normal          => RandomNormal[A](shape, seed)
      case NormalTruncated => RandomNormalTruncated[A](shape, seed)
    }
    val scaled = scale.map(factor => expr * factor.const.cast[A]).getOrElse(expr)
    shift.map(by => scaled + by.const.cast[A]).getOrElse(scaled)
  }
}

object kernels extends AllKernels {
  object syntax extends AllSyntax
}
