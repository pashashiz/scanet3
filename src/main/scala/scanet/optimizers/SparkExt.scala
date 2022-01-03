package scanet.optimizers

import org.apache.spark.sql.types.{Metadata, MetadataBuilder}
import org.apache.spark.sql.{ColumnName, Dataset, Encoder}
import scanet.core.{Convertible, Floating, Monoid, Shape, Tensor, TensorType}
import scanet.math.Dist
import scanet.models.Model
import scanet.optimizers.Optimizer.BuilderState

case class Record[A](features: Array[A], labels: Array[A])
case class TRecord[A: TensorType](features: Tensor[A], labels: Tensor[A])

class DatasetMonoidOps[A: Monoid](val ds: Dataset[Record[A]]) {

  def withShapes(features: Shape, labels: Shape)(implicit
  encoder: Encoder[Record[A]]): Dataset[Record[A]] = {
    def metadata(shape: Shape): Metadata =
      new MetadataBuilder().putLongArray("shape", shape.toLongArray).build()
    ds.select(
      new ColumnName("features").as("features", metadata(features)),
      new ColumnName("labels").as("labels", metadata(labels)))
      .as[Record[A]]
  }

  private def shapeOf(name: String): Shape = {
    val column = ds.schema(name)
    Shape.of(column.metadata.getLongArray("shape"))
  }

  def featuresShape: Shape = shapeOf("features")
  def labelsShape: Shape = shapeOf("labels")
  def shapes: (Shape, Shape) = (featuresShape, labelsShape)

  private def asTRecord[B: TensorType](records: Iterator[(Tensor[B], Tensor[B])]): Array[TRecord[B]] =
    records
      .map((TRecord.apply[B] _).tupled)
      .toArray

  def collectTensors(batch: Int = 1, withPadding: Boolean = false): Array[TRecord[A]] =
    asTRecord(TensorIterator(ds.collect().iterator, shapes, batch, withPadding = withPadding))

  def takeTensors(batch: Int = 1, limit: Int, withPadding: Boolean = false): Array[TRecord[A]] =
    asTRecord(TensorIterator(ds.take(batch * limit).iterator, shapes, batch, withPadding = withPadding))

  def firstTensor(batch: Int = 1, withPadding: Boolean = false): TRecord[A] =
    asTRecord(TensorIterator(ds.take(batch).iterator, shapes, batch, withPadding = withPadding)).head
}

class DatasetFloatingOps[A: Floating: Dist](val ds: Dataset[Record[A]])(
    implicit c: Convertible[Int, A]) {

  def train(model: Model)
      : Optimizer.Builder[A, BuilderState.WithFunc with BuilderState.WithDataset] = {
    Optimizer.minimize[A](model).on(ds)
  }
}

object SparkExt {

  trait AllSyntax {

    implicit def dsMonoidOps[A: Monoid](ds: Dataset[Record[A]]): DatasetMonoidOps[A] =
      new DatasetMonoidOps[A](ds)

    implicit def dsFloatingOps[A: Floating: Dist](ds: Dataset[Record[A]])(
        implicit c: Convertible[Int, A]): DatasetFloatingOps[A] =
      new DatasetFloatingOps[A](ds)
  }

  object syntax extends AllSyntax
}
