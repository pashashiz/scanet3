package org.scanet.optimizers

import org.apache.spark.rdd.RDD
import org.scanet.core.TensorType
import org.scanet.math.{Convertible, Dist, Floating, Numeric}
import org.scanet.models.Model
import org.scanet.optimizers.Optimizer.BuilderState

class RDDOps[A: Numeric: Floating: TensorType: Dist](val rdd: RDD[Array[A]])(
    implicit c: Convertible[Int, A]) {
  def train(
      model: Model): Optimizer.Builder[A, BuilderState.WithFunc with BuilderState.WithDataset] = {
    Optimizer.minimize[A](model).on(rdd)
  }
}

object SparkExt {

  trait Implicits {
    implicit def rddOps[A: Numeric: Floating: TensorType: Dist](rdd: RDD[Array[A]])(
        implicit c: Convertible[Int, A]): RDDOps[A] =
      new RDDOps[A](rdd)
  }

  trait Syntax extends Implicits

  object syntax extends Syntax
}
