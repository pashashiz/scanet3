package org.scanet.optimizers

import org.scanet.core.{Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

// Optimizer(alg = SDG())
//   .minimize(func = x2, init = args) | maximize(...)
//   .on(dataset)
//   .stopAfter(10.epochs)
//   .run(runner = Direct | Spark) | runAsync(...)

case class Optimizer[X: Numeric: TensorType, W: Numeric: TensorType, R: Numeric: TensorType](
            alg: Algorithm,
            batch: Int = Int.MaxValue,
            funcBuilder: TensorFunctionBuilder[X, W, R],
            initArgs: Tensor[W],
            dataset: Dataset[X],
            epochs: Int) {

  def minimize(): Tensor[W] = {
    var arg = initArgs
    var epoch = 0
    var iter = 0
    var it = dataset.iterator
    while (epoch < epochs) {
      if (it.hasNext) {
        iter = iter + 1
        val func = funcBuilder(it.next(batch).const)
        println(s"#: $iter")
        println(s"Result: ${func(arg.const).eval}")
        val grad = func.grad(arg.const).eval
        println(s"Grad  : $grad")
        val step = alg.step(func, arg.const).cast[W]
        println(s"Step  : ${step.eval}")
        arg = (arg.const - step).eval
      } else {
        epoch = epoch + 1
        it = dataset.iterator
      }
    }
    arg
  }
}
