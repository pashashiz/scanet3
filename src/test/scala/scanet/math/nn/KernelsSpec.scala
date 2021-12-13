package scanet.math.nn

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec
import scanet.core.{Shape, Tensor}
import scanet.math.syntax._

class KernelsSpec extends AnyWordSpec with Matchers {

  "conv2D" when {

    val input1 = Tensor.matrix[Double](
      Array(2, 1, 2, 0, 1),
      Array(1, 3, 2, 2, 3),
      Array(1, 1, 3, 3, 0),
      Array(2, 2, 0, 1, 1),
      Array(0, 0, 3, 1, 2))
      .const
      .reshape(1, 5, 5, 1)

    val filters1 = Tensor.matrix[Double](
      Array(2, 3),
      Array(0, 1))
      .const
      .reshape(2, 2, 1, 1)

    "called with Same padding" should {

      /*
      input   = (batch_shape, in_height, in_width, in_channels)          = (1, 5, 5, 1)
      filters = (filter_height, filter_width, in_channels, out_channels) = (2, 2, 1, 1)
      output  = (batch_shape, in_height, in_width, out_channels)         = (1, 5, 5, 1)

      so basically given an image (h, w, 3) and 5 filters
      we can transform (convolve) it into a new image (h1, w2, 5) with more layers

      the convolution means a dot product of a filter with a part of the image
      where filter is multiplied with a part of the image and the resulting tensor
      is summed up, such a filter would move with a given padding + stride
      where stride is basically a step and padding appends zero values to the original matrix
      to make it larger
       */
      "calculate convolution and preserve original HW by auto-padding with stride 1" in {
        val convolved = Conv2D(
          input = input1,
          filters = filters1,
          padding = Same)
        convolved.shape shouldBe Shape(1, 5, 5, 1)
        val expected = Tensor.matrix[Double](
          Array(10.0, 10.0, 6.0, 6.0, 2.0),
          Array(12.0, 15.0, 13.0, 13.0, 6.0),
          Array(7.0, 11.0, 16.0, 7.0, 0.0),
          Array(10.0, 7.0, 4.0, 7.0, 2.0),
          Array(0.0, 9.0, 9.0, 8.0, 4.0))
        convolved.roundAt(2).reshape(5, 5).eval shouldBe expected
      }

      "calculate convolution and preserve HW/2 by auto-padding with stride 2" in {
        val convolved = Conv2D(
          input = input1,
          filters = filters1,
          padding = Same,
          strides = List(2))
        convolved.shape shouldBe Shape(1, 3, 3, 1)
        val expected = Tensor.matrix[Double](
          Array(10.0, 6.0, 2.0),
          Array(7.0, 16.0, 0.0),
          Array(0.0, 9.0, 4.0))
        convolved.roundAt(2).reshape(3, 3).eval shouldBe expected
      }
    }

    "called with Valid padding" should {

      "calculate convolution with no padding and reduce original HW with stride 1" in {
        val convolved = Conv2D(
          input = input1,
          filters = filters1,
          padding = Valid)
        convolved.shape shouldBe Shape(1, 4, 4, 1)
        val expected = Tensor.matrix[Double](
          Array(10.0, 10.0, 6.0, 6.0),
          Array(12.0, 15.0, 13.0, 13.0),
          Array(7.0, 11.0, 16.0, 7.0),
          Array(10.0, 7.0, 4.0, 7.0))
        convolved.roundAt(2).reshape(4, 4).eval shouldBe expected
      }

      "calculate convolution with no padding and reduce original HW with stride 2" in {
        val convolved = Conv2D(
          input = input1,
          filters = filters1,
          padding = Valid,
          strides = List(2))
        convolved.shape shouldBe Shape(1, 2, 2, 1)
        val expected = Tensor.matrix[Double](
          Array(10.0, 6.0),
          Array(7.0, 16.0))
        convolved.roundAt(2).reshape(2, 2).eval shouldBe expected
      }
    }

    "called with Explicit padding" should {

      "calculate convolution if padding fits" in {
        val convolved = Conv2D(
          input = input1,
          filters = filters1,
          padding = Explicit(height = (1, 0), width = (1, 0)),
          strides = List(2))
        convolved.shape shouldBe Shape(1, 3, 3, 1)
        val expected = Tensor.matrix[Double](
          Array(2.0, 2.0, 1.0),
          Array(4.0, 15.0, 13.0),
          Array(6.0, 7.0, 7.0))
        convolved.roundAt(2).reshape(3, 3).eval shouldBe expected
      }
    }

    "calculating gradient in respect to filter" should {

      /*
      given an input matrix
      [
       [2.0  1.0  2.0  0.0  1.0],
       [1.0  3.0  2.0  2.0  3.0],
       [1.0  1.0  3.0  3.0  0.0],
       [2.0  2.0  0.0  1.0  1.0],
       [0.0  0.0  3.0  1.0  2.0]
      ]
      and a filter with shape (2, 2)
      and stride = 1 and no padding
      to calculate a filter gradient for filter i=(0, 0)
      we need to sum up all input elements which would
      contribute to the convolution made by i=(0, 0), which is
      [
       [2.0  1.0  2.0  0.0],
       [1.0  3.0  2.0  2.0],
       [1.0  1.0  3.0  3.0],
       [2.0  2.0  0.0  1.0]
      ]
      we can repeat the same for each filter element
       */
      "return it" in {
        val convolved = Conv2D(
          input = input1,
          filters = filters1,
          padding = Valid)
        val grad = convolved.sum.grad(filters1).returns[Double]
        grad.shape shouldBe filters1.shape
        val expected = Tensor.matrix[Double](
          Array(26.0, 25.0),
          Array(25.0, 27.0))
        grad.roundAt(2).reshape(2, 2).eval shouldBe expected
      }
    }

    "calculating gradient in respect to input" should {

      /*
      the input gradient is calculated in a similar way as for a filter
      given an input matrix with shape (5, 5)
      and a filter
      [
       [2.0, 3.0],
       [0.0, 1.0]
      ]
      if we look at element input(0, 0), the gradient will be filter(0, 0)
      for input(0, 1), the gradient will be filter(0, 0) + filter(0, 1)
      for input(1, 1), the gradient will be filter(0, 0) + filter(0, 1) + filter(1, 0) + filter(1, 1)
       */
      "return it" in {
        val convolved = Conv2D(
          input = input1,
          filters = filters1,
          padding = Valid)
        val grad = convolved.sum.grad(input1).returns[Double]
        grad.shape shouldBe input1.shape
        val expected = Tensor.matrix[Double](
          Array(2.0, 5.0, 5.0, 5.0, 3.0),
          Array(2.0, 6.0, 6.0, 6.0, 4.0),
          Array(2.0, 6.0, 6.0, 6.0, 4.0),
          Array(2.0, 6.0, 6.0, 6.0, 4.0),
          Array(0.0, 1.0, 1.0, 1.0, 1.0))
        grad.roundAt(2).reshape(5, 5).eval shouldBe expected
      }
    }
  }
}
