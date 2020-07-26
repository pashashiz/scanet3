package org.scanet.datasets

import java.io.{DataInputStream, FileInputStream}
import java.net.URL
import java.nio.channels.{Channels, FileChannel}
import java.nio.file.{Files, Path, Paths}
import java.nio.file.StandardOpenOption.WRITE
import java.util.zip.GZIPInputStream

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object MNIST {

  def load(trainingSize: Int, testSize: Int): (RDD[Array[Float]], RDD[Array[Float]]) = {
    val sc = SparkContext.getOrCreate()
    (loadTrainingSet(sc, trainingSize), loadTestSet(sc, testSize))
  }

  def load(sc: SparkContext, trainingSize: Int, testSize: Int): (RDD[Array[Float]], RDD[Array[Float]]) =
    (loadTrainingSet(sc, trainingSize), loadTestSet(sc, testSize))

  def loadTrainingSet(sc: SparkContext, size: Int): RDD[Array[Float]] =
    loadDataSetFrom(sc, images = "train-images-idx3-ubyte.gz", labels = "train-labels-idx1-ubyte.gz", size)

  def loadTestSet(sc: SparkContext, size: Int): RDD[Array[Float]] =
    loadDataSetFrom(sc, images = "t10k-images-idx3-ubyte.gz", labels = "t10k-labels-idx1-ubyte.gz", size)

  def loadDataSetFrom(sc: SparkContext, images: String, labels: String, size: Int): RDD[Array[Float]] = {
    val imagesRdd = sc.makeRDD(loadImages(images, size), 1)
    val labelsRdd = sc.makeRDD(loadLabels(labels, size), 1)
    imagesRdd.join(labelsRdd).map {
      case (_, (images, labels)) => images ++ labels
    }
  }

  def loadImages(name: String, size: Int): Seq[(Int, Array[Float])] = {
    openStream(downloadOrCached(name), stream => {
      require(stream.readInt() == 2051, "wrong MNIST image stream magic number")
      val count = stream.readInt()
      val width = stream.readInt()
      val height = stream.readInt()
      require(size <= count, "passed size is bigger than the actual data set")
      for (i <- 0 until size) yield (i, readImage(stream, height * width))
    })
  }

  def loadLabels(name: String, size: Int): Seq[(Int, Array[Float])] = {
    openStream(downloadOrCached(name), stream => {
      require(stream.readInt() == 2049, "wrong MNIST image stream magic number")
      val count = stream.readInt()
      require(size <= count, "passed size is bigger than the actual data set")
      for (i <- 0 until size) yield (i, readLabel(stream, 10))
    })
  }

  private def openStream[A](path: Path, f: DataInputStream => A) = {
    val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path.toString)))
    try {
      f.apply(stream)
    } finally {
      stream.close()
    }
  }

  def readImage(stream: DataInputStream, size: Int): Array[Float] = {
    Array.range(0, size).map(_ => stream.readUnsignedByte().toFloat / 256f)
  }

  def readLabel(stream: DataInputStream, size: Int): Array[Float] = {
    val m = Array.fill(size)(0f)
    val label = stream.readUnsignedByte()
    m(label) = 1
    m
  }

  private def downloadOrCached(name: String): Path = {
    val resource = Channels.newChannel(new URL(s"http://yann.lecun.com/exdb/mnist/$name").openStream())
    val dir = Paths.get(System.getProperty("user.home"), "mnist")
    Files.createDirectories(dir)
    val file = dir.resolve(name)
    if (!Files.exists(file)) {
      Files.createFile(file)
      FileChannel.open(file, WRITE).transferFrom(resource, 0, Long.MaxValue)
      file
    } else {
      file
    }
  }

}
