package org.scanet.core

import java.nio.ByteBuffer

import org.scanet.core.syntax._
import org.scanet.math.Generator.uniform
import org.scanet.math.Numeric.syntax._
import org.scanet.math.{Convertible, Dist, Generator, Numeric, Random}
import org.scanet.native.{Disposable, NativeTensorOps}
import org.tensorflow.{DataType, Tensor => NativeTensor}

import scala.collection.mutable.ArrayBuffer

class Tensor[A: TensorType](private val ref: TensorRef[A], val view: View) {

  val `type`: TensorType[A] = TensorType[A]
  val buffer: TensorBuffer[A] = TensorBuffer[A](NativeTensorOps.buffer(native), view.originalShape.power)

  def native: NativeTensor[A] = ref.native
  def shape: Shape = view.shape
  def rank: Int = shape.rank
  def power: Int = shape.power
  def isScalar: Boolean = shape.isScalar

  def toScalar: A = {
    require(isScalar, "tensor should be a scalar")
    toArray(0)
  }

  def toArray: Array[A] = {
    val positions = view.positions
    Array.tabulate(positions.length)(i => buffer.read(positions(i)))(TensorType[A].classTag)
  }

  def hasView: Boolean = !view.isIdentity

  /**
    * If any slice or reshape operation was applied before a new tensor
    * will be returned which is based on a compacted byte buffer,
    * otherwise there will be the same tensor
    *
    * @return compacted tensor
    */
  def compact: Tensor[A] = if (hasView) Tensor[A](toArray, shape) else this

  def toByteBuffer: ByteBuffer = compact.buffer.buf

  def toBytes: Array[Byte] = {
    val buffer = toByteBuffer
    val bytes = Array.ofDim[Byte](buffer.capacity())
    buffer.get(bytes)
    buffer.rewind()
    bytes
  }

  def foldLeft[Z](zero: Z)(f: (Z, Tensor[A]) => Z): Z = {
    shape.dims match {
      case Nil => f(zero, this)
      case head::_ =>
        (0 until head).foldLeft(zero)((acc, dim) => {
          f(acc, slice(dim))
        })
    }
  }

  def apply[S1: CanBuildSliceFrom](s1: S1): Tensor[A] = slice(s1)
  def apply[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom](s1: S1, s2: S2): Tensor[A] = slice(s1, s2)
  def apply[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom, S3: CanBuildSliceFrom](s1: S1, s2: S2, s3: S3): Tensor[A] = slice(s1, s2, s3)
  def apply[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom, S3: CanBuildSliceFrom, S4: CanBuildSliceFrom](s1: S1, s2: S2, s3: S3, s4: S4): Tensor[A] = slice(s1, s2, s3, s4)

  def slice[S1: CanBuildSliceFrom](s1: S1): Tensor[A] = slice(Projection(s1))
  def slice[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom](s1: S1, s2: S2): Tensor[A] = slice(Projection(s1, s2))
  def slice[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom, S3: CanBuildSliceFrom](s1: S1, s2: S2, s3: S3): Tensor[A] = slice(Projection(s1, s2, s3))
  def slice[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom, S3: CanBuildSliceFrom, S4: CanBuildSliceFrom](s1: S1, s2: S2, s3: S3, s4: S4): Tensor[A] = slice(Projection(s1, s2, s3, s4))

  def slice(projection: Projection): Tensor[A] = {
    if (projection != view.projection) new Tensor(ref.view, view narrow projection) else this
  }

  def reshape(dims: Int*): Tensor[A] = reshape(Shape(dims: _*))
  def reshape(shape: Shape): Tensor[A] = {
    if (view.shape != shape) new Tensor(ref.view, view reshape shape) else this
  }

  override def toString: String = {
    val sep = if (rank > 1) System.lineSeparator else " "
    s"Tensor[${TensorType[A].show}](shape=${view.shape}):$sep${show()}"
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
      slice(projection).showAll()
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

  implicit def toNativeTensor[A: TensorType](tensor: Tensor[A]): NativeTensor[A] = tensor.ref.native

  def apply[A: TensorType](native: NativeTensor[A]): Tensor[A] = {
    new Tensor(new NativeRef(native), View(Shape.of(native.shape())))
  }

  def apply[A: TensorType](data: Array[A], shape: Shape): Tensor[A] = {
    require(data.length == shape.power,
      s"Shape$shape requires ${shape.power} elements but was passed ${data.length}")
    val size = TensorType[A].coder.sizeOf(data)
    val tensor = Tensor[A](NativeTensorOps.allocateElements[A](shape.toLongArray, size))
    tensor.buffer.write(data)
    tensor
  }

  def fromBytes[A: TensorType](data: Array[Byte], shape: Shape): Tensor[A] = {
    val tensor = Tensor[A](NativeTensorOps.allocateBytes[A](shape.toLongArray, data.length))
    tensor.buffer.writeBytes(data)
    tensor
  }

  def fromBytesUntyped(dataType: DataType, data: Array[Byte], shape: Shape): Tensor[_] =
    fromBytes(data, shape)(TensorType.of(dataType))

  def scalar[A: TensorType](value: A): Tensor[A] = apply(Array(value)(TensorType[A].classTag), Shape())

  def vector(range: Range): Tensor[Int] = apply[Int](range.toArray[Int], Shape(range.length))

  def vector[A: TensorType](array: Array[A]): Tensor[A] = apply(array, Shape(array.length))

  def vector[A: TensorType](elements: A*): Tensor[A] = vector(elements.toArray(TensorType[A].classTag))

  def matrix[A: TensorType](rows: Array[A]*): Tensor[A] = {
    require(rows.nonEmpty, "at least one row is required")
    val rowSizes = rows.toList.map(_.length)
    require(rowSizes.distinct.size == 1, "all rows should have the same length")
    val data = rows.foldLeft(new ArrayBuffer[A](rowSizes.sum))((buffer, row) => buffer ++= row).toArray(TensorType[A].classTag)
    apply(data, Shape(rowSizes.length, rowSizes.head))
  }

  def zeros[A: TensorType: Numeric](shape: Int*): Tensor[A] =
    zeros(Shape(shape.toList))

  def zeros[A: TensorType: Numeric](shape: Shape): Tensor[A] = {
    Tensor(Array.fill(shape.power)(Numeric[A].zero)(TensorType[A].classTag), shape)
  }

  def ones[A: TensorType: Numeric](shape: Int*): Tensor[A] =
    ones(Shape(shape.toList))

  def ones[A: TensorType: Numeric](shape: Shape): Tensor[A] =
    fill(shape)(Numeric[A].one)

  def fill[A: TensorType](shape: Int*)(value: A): Tensor[A] =
    fill(Shape(shape.toList))(value)

  def fill[A: TensorType](shape: Shape)(value: A): Tensor[A] =
    Tensor(Array.fill(shape.power)(value)(TensorType[A].classTag), shape)

  def tabulate[A: TensorType](d1: Int)(f: Int => A): Tensor[A] =
    tabulate(Shape(d1))(idx => f(idx.head))

  def tabulate[A: TensorType](d1: Int, d2: Int)(f: (Int, Int) => A): Tensor[A] =
    tabulate(Shape(d1, d2))(idx => f(idx.head, idx(1)))

  def tabulate[A: TensorType](d1: Int, d2: Int, d3: Int)(f: (Int, Int, Int) => A): Tensor[A] =
    tabulate(Shape(d1, d2, d3))(idx => f(idx.head, idx(1), idx(2)))

  def tabulate[A: TensorType](shape: Shape)(f: List[Int] => A): Tensor[A] = {
    // note: could be optimized, cause indexOf is a reverse operation
    val array = Array.tabulate[A](shape.power)(index => f(shape.indexOf(index)))(TensorType[A].classTag)
    Tensor(array, shape)
  }

  def diag[A: TensorType: Numeric](values: A*): Tensor[A] =
    diag(values.toArray(TensorType[A].classTag))

  def diag[A: TensorType: Numeric](values: Array[A]): Tensor[A] = {
    val zero = Numeric[A].zero
    tabulate(values.length, values.length)((x, y) =>
      if (x == y) values(x) else zero)
  }

  def eye[A: TensorType: Numeric](n: Int): Tensor[A] =
    diag[A](Array.fill(n)(Numeric[A].one)(TensorType[A].classTag))

  def linspace[A: TensorType: Numeric](first: A, last: A, size: Int = 100)(implicit c: Convertible[Int, A]): Tensor[A] = {
    val increment = (last - first) / c.convert(size - 1)
    tabulate(size)(i => first plus (increment * i))
  }

  def range(range: Range): Tensor[Int] = Tensor.range[Int](range.start, range.end, 1)

  def range[A: TensorType: Numeric](start: A, end: A, step: A, inclusive: Boolean = false)(implicit c1: Convertible[A, Int], c2: Convertible[Int, A]): Tensor[A] = {
    val sizeAprox = c1.convert((end - start) / step) + 1
    val endAprox = start.plus(step * (sizeAprox - 1))
    val size =
      if (endAprox < end || inclusive && (endAprox === end)) {
        sizeAprox
      } else {
        sizeAprox - 1
      }
    tabulate(size.toInt)(i => start plus (step * i))
  }

  def rand[A: TensorType: Numeric: Dist](shape: Shape, gen: Generator = uniform): Tensor[A] = {
    val (_, arr) = Random[A](gen).next(shape.power)(TensorType[A].classTag, Dist[A])
    Tensor[A](arr, shape)
  }
}

sealed trait TensorRef[A] {

  def native: NativeTensor[A]

  def view: TensorRef[A]
}

class NativeRef[A: TensorType](val native: NativeTensor[A]) extends Disposable(() => native.close()) with TensorRef[A] {

  override def view: TensorRef[A] = new ParentRef[A](this)
}

// capture reference to origin TensorRef to prevent it from being GCed (together with its native tensor)
class ParentRef[A: TensorType](private val origin: TensorRef[A]) extends TensorRef[A] {

  val native: NativeTensor[A] = origin.native

  val view: TensorRef[A] = this
}
