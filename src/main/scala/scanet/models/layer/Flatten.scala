package scanet.models.layer

import scanet.core.syntax._
import scanet.core.{Expr, Floating, Shape}

/** A layer which flattens the input tensor of any shape to 2 dims matrix.
  *
  * Given an input tensor `Shape(N, H, W, C)`, after flattening we will get `Shape(N, H*W*C)`
  */
case object Flatten extends NotTrainableLayer {

  override def build_[E: Floating](input: Expr[E]): Expr[E] = {
    val shape = input.shape
    require(shape.rank >= 2, s"rank should be >= 2, but was ${shape.rank}")
    val batch = shape(0)
    val features = (shape << 1).power
    input.reshape(batch, features)
  }

  override def outputShape(input: Shape): Shape = Shape(input.head, input.tail.power)
}
