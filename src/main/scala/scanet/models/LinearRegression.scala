package scanet.models

import scanet.models.Activation.Identity
import scanet.models.Initializer.{GlorotUniform, Zeros}
import scanet.models.Regularization.Zero
import scanet.models.layer.Dense

object LinearRegression {

  /** Ordinary least squares Linear Regression.
    *
    * LinearRegression fits a linear model with coefficients `w = (w1, …, wn)`
    * to minimize the residual sum of squares between the observed targets in the dataset,
    * and the targets predicted by the linear approximation.
    *
    * Model always has only one output
    *
    * That is equivalent to `layer.Dense(1, Identity, reg, bias)`
    */
  def apply(
      reg: Regularization = Zero,
      bias: Boolean = true,
      kernelInitializer: Initializer = Zeros,
      biasInitializer: Initializer = Zeros): Model =
    Dense(
      outputs = 1,
      activation = Identity,
      reg = reg,
      bias = bias,
      kernelInitializer = kernelInitializer,
      biasInitializer = biasInitializer)
}
