package scanet.datasets

import org.apache.spark.sql.{Dataset, SparkSession}
import scanet.core.{Shape, Using}
import scanet.optimizers.Record
import scanet.optimizers.syntax._
import scanet.core.syntax._

import java.io.DataInputStream
import java.net.URL
import java.nio.channels.{Channels, FileChannel}
import java.nio.file.StandardOpenOption.WRITE
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPInputStream

object MNIST {

  def load(trainingSize: Int = 60000, testSize: Int = 10000)(implicit
  spark: SparkSession): (Dataset[Record[Float]], Dataset[Record[Float]]) =
    (loadTrainingSet(trainingSize), loadTestSet(testSize))

  def loadTrainingSet(size: Int)(implicit spark: SparkSession): Dataset[Record[Float]] =
    loadDataSetFrom(
      images = "train-images-idx3-ubyte.gz",
      labels = "train-labels-idx1-ubyte.gz",
      size)

  def loadTestSet(size: Int)(implicit spark: SparkSession): Dataset[Record[Float]] =
    loadDataSetFrom(
      images = "t10k-images-idx3-ubyte.gz",
      labels = "t10k-labels-idx1-ubyte.gz",
      size)

  def loadDataSetFrom(
      images: String,
      labels: String,
      size: Int)(implicit spark: SparkSession): Dataset[Record[Float]] = {
    import spark.implicits._
    def read(path: String, via: DataInputStream => Seq[(Int, Array[Float])]) = {
      spark.sparkContext.binaryFiles(downloadOrCached(path).toAbsolutePath.toString, 1)
        .flatMap {
          case (_, portableStream) =>
            Using.resource(new DataInputStream(new GZIPInputStream(portableStream.open())))(via(_))
        }
    }
    val imagesRdd = read(images, readImages(_, size))
    val labelsRdd = read(labels, readLabels(_, size))
    imagesRdd.join(labelsRdd)
      .map {
        case (_, (images, labels)) => Record(images, labels)
      }
      .toDS()
      // todo: Shape(28, 28)
      .withShapes(Shape(28 * 28), Shape(10))
  }

  def readImages(stream: DataInputStream, size: Int): Seq[(Int, Array[Float])] = {
    require(stream.readInt() == 2051, "wrong MNIST image stream magic number")
    val count = stream.readInt()
    val width = stream.readInt()
    val height = stream.readInt()
    require(size <= count, "passed size is bigger than the actual data set")
    for (i <- 0 until size) yield (i, readImage(stream, height * width))
  }

  def readLabels(stream: DataInputStream, size: Int): Seq[(Int, Array[Float])] = {
    require(stream.readInt() == 2049, "wrong MNIST image stream magic number")
    val count = stream.readInt()
    require(size <= count, "passed size is bigger than the actual data set")
    for (i <- 0 until size) yield (i, readLabel(stream, 10))
  }

  def readImage(stream: DataInputStream, size: Int): Array[Float] = {
    Array.range(0, size).map(_ => (stream.readUnsignedByte().toFloat + 1f) / 256f)
  }

  def readLabel(stream: DataInputStream, size: Int): Array[Float] = {
    val m = Array.fill(size)(0f)
    val label = stream.readUnsignedByte()
    m(label) = 1
    m
  }

  private def downloadOrCached(name: String): Path = {
    val dir = Paths.get(System.getProperty("user.home"), ".scanet", "cache", "mnist")
    Files.createDirectories(dir)
    val file = dir.resolve(name)
    if (!Files.exists(file)) {
      val resource = Channels.newChannel(
        new URL(s"https://ossci-datasets.s3.amazonaws.com/mnist/$name").openStream())
      Files.createFile(file)
      FileChannel.open(file, WRITE).transferFrom(resource, 0, Long.MaxValue)
      file
    } else {
      file
    }
  }
}
