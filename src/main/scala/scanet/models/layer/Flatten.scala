package scanet.models.layer

import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.math.syntax.zeros
import scanet.core.syntax._

import scala.collection.immutable.Seq

/** A layer which flattens the input tensor of any shape to 2 dims matrix.
  *
  * Given an input tensor `Shape(N, H, W, C)`, after flattening we will get `Shape(N, H*W*C)`
  */
case object Flatten extends Layer {

  override def build[E: Floating](x: Expr[E], weights: OutputSeq[E]): Expr[E] = {
    require(weights.isEmpty, "Flatten layer does not require weights")
    val shape = x.shape
    require(shape.rank >= 2, s"rank should be >= 2, but was ${shape.rank}")
    val batch = shape(0)
    val features = (shape << 1).power
    x.reshape(batch, features)
  }

  override def penalty[E: Floating](weights: OutputSeq[E]): Expr[E] = zeros[E](Shape())

  override def outputShape(input: Shape): Shape = Shape(input.power)

  override def weightsCount: Int = 0

  override def weightsShapes(input: Shape): Seq[Shape] = Seq.empty
}
