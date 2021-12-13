package scanet.optimizers

import org.apache.spark.rdd.RDD
import scanet.core.{Convertible, Floating}
import scanet.math.Dist
import scanet.models.Model
import scanet.optimizers.Optimizer.BuilderState

class RDDOps[A: Floating: Dist](val rdd: RDD[Array[A]])(
    implicit c: Convertible[Int, A]) {
  def train(
      model: Model): Optimizer.Builder[A, BuilderState.WithFunc with BuilderState.WithDataset] = {
    Optimizer.minimize[A](model).on(rdd)
  }
}

object SparkExt {

  trait AllSyntax {
    implicit def rddOps[A: Floating: Dist](rdd: RDD[Array[A]])(
        implicit c: Convertible[Int, A]): RDDOps[A] =
      new RDDOps[A](rdd)
  }

  object syntax extends AllSyntax
}
