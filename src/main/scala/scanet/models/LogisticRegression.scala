package scanet.models

import scanet.models.Activation.Sigmoid
import scanet.models.Initializer.Zeros
import scanet.models.Regularization.Zero
import scanet.models.layer.Dense

object LogisticRegression {

  /** Binary Logistic Regression
    *
    * Similar to Linear Regression but applies a logistic function on top
    * to model a binary dependent variable. Hence, the result is a probability in range `[0, 1]`.
    *
    * Model always has only one output
    *
    * That is equivalent to `layer.Dense(1, Sigmoid, reg, bias)`
    */
  def apply(
      reg: Regularization = Zero,
      bias: Boolean = true,
      kernelInitializer: Initializer = Zeros,
      biasInitializer: Initializer = Zeros): Model =
    Dense(
      outputs = 1,
      activation = Sigmoid,
      reg = reg,
      bias = bias,
      kernelInitializer = kernelInitializer,
      biasInitializer = biasInitializer)
}
