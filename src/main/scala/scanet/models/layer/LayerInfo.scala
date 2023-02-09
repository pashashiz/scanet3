package scanet.models.layer

import scanet.core.Shape

import scala.annotation.tailrec
import scala.collection.immutable.Seq

case class LayerInfo(name: String, weights: Seq[Shape], output: Shape) {

  private def group[A](input: Seq[A]): (Seq[A], Int) = {
    def repeated(size: Int): Boolean = {
      if (input.size % size == 0) {
        val groups = input.sliding(size, size).toList
        groups.distinct.size == 1
      } else {
        false
      }
    }
    @tailrec
    def scan(size: Int): (Seq[A], Int) = {
      if (input.size >= size.toDouble / 2) {
        if (repeated(size))
          (input.take(size), input.size / size)
        else
          scan(size + 1)
      } else {
        (input, 1)
      }
    }
    scan(1)
  }

  private def groupConcat[A](input: Seq[A]): String = {
    val (elements, size) = group(input)
    val value = elements.map(_.toString).mkString(", ")
    if (size > 1) s"[$value]x$size" else value
  }

  def toRow: Seq[String] = {
    val weightsStr = groupConcat(weights)
    val params = groupConcat(weights.map(_.power))
    val outputStr = ("_" +: output.tail.dims.map(_.toString)).mkString("(", ",", ")")
    Seq(name, weightsStr, params, outputStr)
  }
}
