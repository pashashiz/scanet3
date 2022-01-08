package scanet.models.layer

import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.math.syntax.zeros
import scanet.core.syntax._

import scala.collection.immutable.Seq

case class Flatten() extends Layer {

  override def build[E: Floating](x: Expr[E], weights: OutputSeq[E]): Expr[E] = {
    require(weights.isEmpty, "Flatten layer does not require weights")
    val shape = x.shape
    require(shape.rank >= 2, s"rank should be >= 2, but was ${shape.rank}")
    val batch = shape(0)
    val features = (shape << 1).power
    // Cannot reshape a tensor with 50176000 elements to shape [1000,784] (784000 elements) for '{{node Reshape_1}} = Reshape[T=DT_FLOAT, Tshape=DT_INT32](MaxPoolV2_1, new_shape_1)' with input shapes: [1000,28,28,64], [2] and with input tensors computed as partial shapes: input[1] = [1000,784].
    x.reshape(batch, features)
  }

  override def penalty[E: Floating](weights: OutputSeq[E]): Expr[E] = zeros[E](Shape())

  override def outputShape(input: Shape): Shape = Shape(input.power)

  override def weightsCount: Int = 0

  override def weightsShapes(input: Shape): Seq[Shape] = Seq.empty
}
