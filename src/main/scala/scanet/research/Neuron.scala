package scanet.research

import scanet.core.Params.Weights
import scanet.core.{Params, Shape, Tensor}
import scanet.models.Activation.Identity
import scanet.models.Loss.MeanSquaredError
import scanet.models.layer.Dense
import scanet.syntax._

import scala.collection.immutable.Seq

object Neuron {

  def main(args: Array[String]): Unit = {
    val neuron = Dense(1, Identity, bias = false)
    // x: dataset, where each record consists of set of features
    // for example, lets take a person and try to predict how much money
    // he has on a bank account given the time he works each week and age
    // features: (week working hours, age)
    val x = Tensor.matrix(Array(50f, 25f), Array(5f, 18f))
    // y: expected values for bank account
    val y = Tensor.matrix(Array(40f), Array(10f))
    // we need to take some random weights with expected shape
    println(neuron.outputShape(Shape(2)))
    // (1, 3) -> 3 = 2 + 1 -> 1 is bias
    val w = Tensor.matrix(Array(0.6f, 0.35f, 0.9f))
    val params = Params(Weights -> w)
    // to make a prediction we need to run a forward pass
    val result = neuron.result[Float].compile
    // (0.7 * 50 + 0.5 * 25 + 1 * 1)
    println(result(x, params))
    val paramsExpr = params.mapValues(_.const)
    // let's calculate prediction error (loss)
    val loss = MeanSquaredError.build(neuron.build(x.const, paramsExpr)._1, y.const)
    println(loss.eval)
    // let's calculate a gradient
    val grads = loss.grad(paramsExpr).returns[Float]
    println(grads.eval)
    // now we can subtract a fraction of a gradient from weights
    // and next time loss should b smaller which means more accurate prediction
  }
}
