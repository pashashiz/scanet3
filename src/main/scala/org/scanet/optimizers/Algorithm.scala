package org.scanet.optimizers

import org.scanet.core.{Output, Tensor, TensorType}
import org.scanet.math.Numeric

trait Algorithm {

  def delta[W: TensorType : Numeric, R: TensorType : Numeric](model: Output[R], arg: Output[W]): Delta
}

case class Delta(delta: Output[Float], metadata: Metadata)
object Delta {
  def apply(delta: Output[Float], metaVars: Variable[Float]*): Delta = {
    Delta(delta, Metadata(metaVars))
  }
}

case class Metadata(vars: Seq[Variable[Float]]) {

  def feed: Map[Output[_], Tensor[_]] = Map.from(vars.map(_.feed))

  def outputs: Seq[Output[Float]] = vars.map(v => Some(v.output).get)

  def next(result: Seq[Tensor[Float]]): Metadata =
    Metadata(vars.zip(result).map { case (v, r) => v.next(r) })
}

case class Variable[A: TensorType : Numeric] private(
                                                      placeholder: Output[A],
                                                      private[optimizers] val output: Output[A],
                                                      private[optimizers] val prev: Tensor[A]
                                                    ) {

  def output(o: Output[A]): Variable[A] = copy(output = o)

  def feed: (Output[_], Tensor[_]) = (placeholder, prev)

  def next(value: Tensor[A]): Variable[A] = copy(prev = value)
}
object Variable {

  def init[A: TensorType : Numeric](value: Tensor[A]): Variable[A] =
    Variable[A](placeholder[A](value.shape), output = null, value)
}