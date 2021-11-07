package scanet.math.grad

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec
import scanet.core.{Expr, OutputSeq, Tensor}
import scanet.core.Session.withing
import scanet.core.Tensor.scalar
import scanet.math.syntax._

import scala.collection.JavaConverters._
import scala.collection.immutable.Seq

class GradSpec extends AnyWordSpec with Matchers {

  "gradient" when {

    "taken in respect to multiple inputs" should {

      "work" in {
        val a = 3f.const.as("a")
        val b = 4f.const.as("b")
        val f = (a * b).pow(3)
        val grads: OutputSeq[Float] = f.grad(Seq(a, b)).returns[Float]
        grads.eval should be(Seq(scalar(1728f), scalar(1296f)))
      }

      "reuse common operations of a computation graph" in {
        val a = 3f.const.as("a")
        val b = 4f.const.as("b")
        val f = (a * b).pow(3)
        val ga = f.grad(a).returns[Float].as("grad_a")
        val gb = f.grad(b).returns[Float].as("grad_b")
        // in our case both grads will reuse
        // 3 * (a * b)^2 sub-graph
        (ga, gb).eval should be((scalar(1728f), scalar(1296f)))
        withing(session => {
          val graph = session.toGraph(Seq(ga, gb))
          val ops = graph.operations().asScala
          ops should have size 11
        })
      }
    }
  }

  "gradient" should {

    "compute second partial derivative" in {
      val x = 5f.const.as("x")
      val y = x.sqr.as("y")
      val dy_dx = y.grad(x).returns[Float]
      val dy_dx2 = dy_dx.grad(x).returns[Float]
      dy_dx2.eval should be(Tensor.scalar(2f))
    }

    "compute successive (mixed) partial derivative" in {
      val x = 5f.const.as("x")
      val y = 2f.const.as("y")
      val z = (x.sqr * y.pow(3) + 3f.const * y + x + 5f.const).as("z")
      val dz_dx = z.grad(x).returns[Float] // 2x * y^3 + 1
      val dz_dx_dy = dz_dx.grad(y).returns[Float] // 6x * y^2
      dz_dx_dy.eval should be(Tensor.scalar(120f))
    }

    "compute Jacobian matrix" in {

      // given f(x, y)
      // we can calc gradients [df/dx, df/dy]
      // and if we calc Jacobian for those
      // | d2f/dx_dx d2f/dx_dy |
      // | d2f/dy_dx d2f/dy_dy |
      // we will end up with Hessian

      // NOTE: Postponed case requires lots of work and will not bring much value as of now
      def jacobian[A](outputs: Expr[A], inputs: Expr[A]): Expr[A] = {
        // To implement that in vectorized form we need to
        // implement: flatten, gather and for loop kernels.
        // So we could take a tensor of outputs
        // flatten it, and for each i-th element gather it and find a gradient
        // in regards to the inputs.
        // At the end we need to join it back to a single tensor.
        // See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/parallel_for/gradients.py#L24
        // For neural network we can then easily find a hessian matrix
        // val grads = loss.gradient(w)
        // val hessian = jacobian(grads, w)
        ???
      }
    }
  }
}
