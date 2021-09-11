package scanet.optimizers

import org.apache.spark.rdd.RDD
import scanet.core.TensorType
import scanet.math.{Convertible, Dist, Floating, Numeric}
import scanet.models.Model
import scanet.optimizers.Optimizer.BuilderState

class RDDOps[A: Numeric: Floating: TensorType: Dist](val rdd: RDD[Array[A]])(
    implicit c: Convertible[Int, A]) {
  def train(
      model: Model): Optimizer.Builder[A, BuilderState.WithFunc with BuilderState.WithDataset] = {
    Optimizer.minimize[A](model).on(rdd)
  }
}

object SparkExt {

  trait AllSyntax {
    implicit def rddOps[A: Numeric: Floating: TensorType: Dist](rdd: RDD[Array[A]])(
        implicit c: Convertible[Int, A]): RDDOps[A] =
      new RDDOps[A](rdd)
  }

  object syntax extends AllSyntax
}
