package org.scanet.optimizers

import org.scanet.core.{Output, TensorType}
import org.scanet.math.Numeric

// Optimizer(alg = SDG())
//   .minimize(func = x2, init = args) | maximize(...)
//   .on(dataset)
//   .stopAfter(10.epochs)
//   .run(runner = Direct | Spark) | runAsync(...)

case class Optimizer(alg: Algorithm, batch: Int = 0) {

  def minimize[A1: Numeric: TensorType, A2: Numeric: TensorType, B: Numeric: TensorType](func: TensorFunctionBuilder[A1, A2, B]): Output[A2] = {


    ???
  }
}
