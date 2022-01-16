package scanet.models.layer

import scanet.core.syntax._
import scanet.core.{Expr, Floating, OutputSeq, Shape}

/** A layer which flattens the input tensor of any shape to 2 dims matrix.
  *
  * Given an input tensor `Shape(N, H, W, C)`, after flattening we will get `Shape(N, H*W*C)`
  */
case object Flatten extends StatelessLayer {

  override def build[E: Floating](x: Expr[E], weights: OutputSeq[E]): Expr[E] = {
    require(weights.isEmpty, "Flatten layer does not require weights")
    val shape = x.shape
    require(shape.rank >= 2, s"rank should be >= 2, but was ${shape.rank}")
    val batch = shape(0)
    val features = (shape << 1).power
    x.reshape(batch, features)
  }

  override def outputShape(input: Shape): Shape = Shape(input.power)
}
