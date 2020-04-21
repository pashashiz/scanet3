package org.scanet.core

import org.scanet.math
import org.scanet.math.Generator.uniform
import org.scanet.math.{Dist, Generator, Random}
import org.scanet.native.{Disposable, NativeTensorOps}
import org.tensorflow.{Tensor => NativeTensor}

import scala.collection.mutable.ArrayBuffer
import scala.{specialized => sp}
import org.scanet.syntax.core._

class Tensor[@sp A: math.Numeric](val native: NativeTensor[A], val view: View) extends Disposable(() => native.close()) {

  val buffer: Buffer[A] = NativeTensorOps.buffer(native)

  def shape: Shape = view.shape
  def rank: Int = shape.rank
  def power: Int = shape.power
  def isScalar: Boolean = shape.isScalar

  def toScalar: A = {
    require(isScalar, "tensor should be a scalar")
    toArray(0)
  }

  override def finalize(): Unit = {
    println("final")
  }

  def toArray: Array[A] = {
    val positions = view.positions
    Array.tabulate(positions.length)(i => buffer.get(positions(i)))(math.Numeric[A].classTag)
  }

  def foldLeft[Z](zero: Z)(f: (Z, Tensor[A]) => Z): Z = {
    shape.dims match {
      case Nil => f(zero, this)
      case head::_ =>
        (0 until head).foldLeft(zero)((acc, dim) => {
          f(acc, get(dim))
        })
    }
  }

  def apply[S1: CanBuildSliceFrom](s1: S1): Tensor[A] = get(s1)
  def apply[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom](s1: S1, s2: S2): Tensor[A] = get(s1, s2)
  def apply[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom, S3: CanBuildSliceFrom](s1: S1, s2: S2, s3: S3): Tensor[A] = get(s1, s2, s3)
  def apply[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom, S3: CanBuildSliceFrom, S4: CanBuildSliceFrom](s1: S1, s2: S2, s3: S3, s4: S4): Tensor[A] = get(s1, s2, s3, s4)

  def get[S1: CanBuildSliceFrom](s1: S1): Tensor[A] = get(Projection(s1))
  def get[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom](s1: S1, s2: S2): Tensor[A] = get(Projection(s1, s2))
  def get[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom, S3: CanBuildSliceFrom](s1: S1, s2: S2, s3: S3): Tensor[A] = get(Projection(s1, s2, s3))
  def get[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom, S3: CanBuildSliceFrom, S4: CanBuildSliceFrom](s1: S1, s2: S2, s3: S3, s4: S4): Tensor[A] = get(Projection(s1, s2, s3, s4))

  def get(projection: Projection): Tensor[A] = new Tensor(native, view narrow projection)

  def reshape(dims: Int*): Tensor[A] = reshape(Shape(dims: _*))
  def reshape(shape: Shape): Tensor[A] = new Tensor(native, view reshape shape)

  override def toString: String = {
    val sep = if (rank > 1) System.lineSeparator else " "
    s"Tensor[${math.Numeric[A].show}](shape=${view.shape}):$sep${show()}"
  }

  def show(): String = {
    val limits = (0 until rank).reverse.map {
      case 0 => 20
      case 1 => 3
      case _ => 2
    }
    show(limits: _*)
  }

  def show(limits: Int*): String = {
    require(limits.size == rank,
      s"specified limits for ${limits.size} dimensions, but tensor has rank $rank")
    if (view.isScalar) {
      toScalar.toString
    } else {
      val projection = Projection.of(shape) narrow
        Projection(limits.map(max => (0 until max).build))
      get(projection).showAll()
    }
  }

  private def showAll(baseShift: String = ""): String = {
    if (isScalar) {
      toScalar.toString
    } else {
      val nl = System.lineSeparator
      val children = foldLeft("")((acc, next) => {
        val sep = if (acc.isEmpty) "" else if (rank > 1) s",$nl" else ", "
        val shift = if (rank > 1) baseShift + "  " else ""
        acc + sep + shift + next.showAll(shift)
      })
      if (rank > 1) {
        s"[$nl$children$nl$baseShift]"
      } else {
        s"[$children]"
      }
    }
  }

  override def hashCode(): Int = view.hashCode() + toArray.foldLeft(1)((acc, a) => 31 * acc + a.hashCode())

  override def equals(obj: Any): Boolean = obj match {
    case other: Tensor[A] =>
      other.view.shape == view.shape &&
        (other.toArray sameElements toArray)
    case _ => false
  }
}

object Tensor {

  implicit def toNativeTensor[@sp A: math.Numeric](tensor: Tensor[A]): NativeTensor[A] = tensor.native

  def apply[@sp A: math.Numeric](native: NativeTensor[A]): Tensor[A] = {
    // note: pre-initialized variables to overcome @sp issue https://github.com/scala/bug/issues/4511
    new Tensor(native, View(Shape.of(native.shape())))
  }

  def apply[@sp A: math.Numeric](data: Buffer[A], shape: Shape): Tensor[A] = {
    val tensor = Tensor[A](NativeTensorOps.allocate[A](shape))
    tensor.buffer.put(data)
    tensor.buffer.rewind
    tensor
  }

  def apply[@sp A: math.Numeric](data: Array[A], shape: Shape): Tensor[A] = {
    require(data.length == shape.power,
      s"Shape$shape requires ${shape.power} elements but was passed ${data.length}")
    apply(Buffer.wrap(data), shape)
  }

  def scalar[@sp A: math.Numeric](value: A): Tensor[A] = apply(Array(value)(math.Numeric[A].classTag), Shape())

  def vector(range: Range): Tensor[Int] = apply[Int](range.toArray[Int], Shape(range.length))

  def vector[@sp A: math.Numeric](array: Array[A]): Tensor[A] = apply(array, Shape(array.length))

  def vector[@sp A: math.Numeric](elements: A*): Tensor[A] = vector(elements.toArray(math.Numeric[A].classTag))

  def matrix[@sp A: math.Numeric](rows: Array[A]*): Tensor[A] = {
    require(rows.nonEmpty, "at least one row is required")
    val rowSizes = rows.toList.map(_.length)
    require(rowSizes.distinct.size == 1, "all rows should have the same length")
    val data = rows.foldLeft(new ArrayBuffer[A](rowSizes.sum))((buffer, row) => buffer ++= row).toArray(math.Numeric[A].classTag)
    apply(data, Shape(rowSizes.length, rowSizes.head))
  }

  def zeros[@sp A: math.Numeric](shape: Int*): Tensor[A] =
    zeros(Shape(shape.toList))

  def zeros[@sp A: math.Numeric](shape: Shape): Tensor[A] =
    Tensor(Buffer.allocate[A](shape.power), shape)

  def ones[@sp A: math.Numeric](shape: Int*): Tensor[A] =
    ones(Shape(shape.toList))

  def ones[@sp A: math.Numeric](shape: Shape): Tensor[A] =
    fill(shape)(math.Numeric[A].one)

  def fill[@sp A: math.Numeric](shape: Int*)(value: A): Tensor[A] =
    fill(Shape(shape.toList))(value)

  def fill[@sp A: math.Numeric](shape: Shape)(value: A): Tensor[A] =
    Tensor(Buffer.tabulate[A](shape.power)(_ => value), shape)

  def tabulate[@sp A: math.Numeric](d1: Int)(f: Int => A): Tensor[A] =
    tabulate(Shape(d1))(idx => f(idx.head))

  def tabulate[@sp A: math.Numeric](d1: Int, d2: Int)(f: (Int, Int) => A): Tensor[A] =
    tabulate(Shape(d1, d2))(idx => f(idx.head, idx(1)))

  def tabulate[@sp A: math.Numeric](d1: Int, d2: Int, d3: Int)(f: (Int, Int, Int) => A): Tensor[A] =
    tabulate(Shape(d1, d2, d3))(idx => f(idx.head, idx(1), idx(2)))

  def tabulate[@sp A: math.Numeric](shape: Shape)(f: List[Int] => A): Tensor[A] = {
    // note: could be optimized, cause indexOf is a reverse operation
    val buffer = Buffer.tabulate[A](shape.power)(index => f(shape.indexOf(index)))
    Tensor(buffer, shape)
  }

  def diag[@sp A: math.Numeric](values: A*): Tensor[A] =
    diag(values.toArray(math.Numeric[A].classTag))

  def diag[@sp A: math.Numeric](values: Array[A]): Tensor[A] = {
    val zero = math.Numeric[A].zero
    tabulate(values.length, values.length)((x, y) =>
      if (x == y) values(x) else zero)
  }

  def eye[@sp A: math.Numeric](n: Int): Tensor[A] =
    diag[A](Array.fill(n)(math.Numeric[A].one)(math.Numeric[A].classTag))

  def linspace[@sp A: math.Numeric](first: A, last: A, size: Int = 100): Tensor[A] = {
    val increment = (last - first) / (size - 1)
    tabulate(size)(i => first plus (increment * i))
  }

  def range(range: Range): Tensor[Int] = Tensor.range[Int](range.start, range.end, 1)

  def range[@sp A: math.Numeric](start: A, end: A, step: A, inclusive: Boolean = false): Tensor[A] = {
    val sizeAprox = ((end - start) / step).toInt + 1
    val endAprox = start.plus(step * (sizeAprox - 1))
    val size =
      if (endAprox < end || inclusive && (endAprox === end)) {
        sizeAprox
      } else {
        sizeAprox - 1
      }
    tabulate(size.toInt)(i => start plus (step * i))
  }

  def rand[@sp A: math.Numeric: Dist](shape: Shape, gen: Generator = uniform): Tensor[A] = {
    val (_, arr) = Random[A](gen).next(shape.power)(math.Numeric[A].classTag, Dist[A])
    Tensor[A](arr, shape)
  }
}
