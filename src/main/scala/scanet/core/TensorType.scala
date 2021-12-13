package scanet.core

import org.tensorflow.proto.framework.DataType
import org.tensorflow.types._
import org.tensorflow.types.family.TType
import simulacrum.{op, typeclass}

import scala.reflect.ClassTag

@typeclass trait TensorType[A] {
  def tag: DataType
  def jtag: Class[_ <: TType]
  def code: Int = tag.getNumber
  def classTag: ClassTag[A]
  def show: String = classTag.toString()
  def codec: TensorCodec[A]
}

@typeclass trait Eq[A] extends TensorType[A] {
  @op("===", alias = true)
  def eqv(x: A, y: A): Boolean
  @op("=!=", alias = true)
  def neqv(x: A, y: A): Boolean = !eqv(x, y)
}

@typeclass trait Order[A] extends Eq[A] {
  def compare(x: A, y: A): Int
  @op(">", alias = true)
  def gt(x: A, y: A): Boolean = compare(x, y) > 0
  @op(">=", alias = true)
  def gte(x: A, y: A): Boolean = compare(x, y) >= 0
  @op("<", alias = true)
  def lt(x: A, y: A): Boolean = compare(x, y) < 0
  @op("<=", alias = true)
  def lte(x: A, y: A): Boolean = compare(x, y) <= 0
  override def eqv(x: A, y: A): Boolean = compare(x, y) == 0
}

@typeclass trait Semiring[A] extends TensorType[A] {

  /** Add two elements.
    *
    * Requirements:
    * - `left` and `right` should have the same dimensions
    * - or one of the tensors should have shape which includes the other
    *
    * If both elements are scalars a simple scalar addition is done:
    *
    * For numbers
    * {{{2 plus 3 should be(5)}}}
    *
    * For tensors
    * {{{
    * val a = Tensor.matrix(
    *   Array(1, 2),
    *   Array(1, 2))
    * val b = Tensor.vector(1, 2)
    * val c = Tensor.matrix(
    *   Array(2, 4),
    *   Array(2, 4))
    * (a.const plus b.const).eval should be(c)
    * }}}
    *
    * @param left side
    * @param right side
    * @tparam B type which can be converted into output
    * @return a result of addition
    */
  // todo: figure out why + operator is not resolved
  @op("+", alias = true)
  def plus[B](left: A, right: B)(implicit c: Convertible[B, A]): A

  /** Multiply 2 elements.
    *
    * If both elements are scalars a simple scalar multiplication is done:
    *
    * For numbers
    * {{{2 * 3 should be(6)}}}
    *
    * For tensors
    * {{{(2.const * 3.const).eval should be(Tensor.scalar(6))}}}
    *
    * If a left element is a scalar and right is a vector, each element in a vector will be multiplied by a scalar
    * {{{(2.const * Tensor.vector(1, 2, 3).const).eval should be(Tensor.vector(2, 4, 6))}}}
    *
    * If a left element is a vector and right is a scalar - an error will be raised cause dimensions are incompatible
    *
    * If a left element is a vector and right is a matrix, a vector will be reshaped into a matrix
    * and a matrix multiplication will be done, the result will be squeezed to a vector though
    * {{{
    *  val a = Tensor.vector(1, 2, 3)
    *  val b = Tensor.matrix(
    *       Array(1, 2),
    *       Array(1, 2),
    *       Array(1, 2))
    * (a.const * b.const).eval should be(Tensor.vector(6, 12))
    * }}}
    *
    * If two matrices are multiplied the regular matmul is done:
    * {{{
    * val a = Tensor.matrix(
    *       Array(1, 2, 3),
    *       Array(1, 2, 3))
    * val b = Tensor.matrix(
    *       Array(1, 2),
    *       Array(1, 2),
    *       Array(1, 2))
    * val c = Tensor.matrix(
    *       Array(6, 12),
    *       Array(6, 12))
    * (a.const * b.const).eval should be(c)
    * }}}
    *
    * NOTE: N-dim tensors are not supported yet, but that will be done in the future
    *
    * @param left side
    * @param right side
    * @tparam B type which can be converted into output
    * @return a result of multiplication
    */
  @op("*", alias = true)
  def multiply[B](left: A, right: B)(implicit c: Convertible[B, A]): A
}

@typeclass trait Rng[A] extends Semiring[A] {
  def zero: A
  @op("-", alias = true)
  def minus[B](left: A, right: B)(implicit c: Convertible[B, A]): A
  // todo: figure out why unary_- operator is not resolved
  @op("unary_-", alias = true)
  def negate(a: A): A
}

@typeclass trait Rig[A] extends Semiring[A] {
  def one: A
}

@typeclass trait Ring[A] extends Rng[A] with Rig[A] {}

@typeclass trait Field[A] extends Ring[A] {
  @op("/", alias = true)
  def div[B](left: A, right: B)(implicit c: Convertible[B, A]): A
}

@typeclass trait Numeric[A] extends Field[A] with Order[A] {}

@typeclass trait Floating[A] extends Numeric[A]

@typeclass trait Logical[A] extends TensorType[A] {
  @op("&&", alias = true)
  def and(x: A, y: A): Boolean
  @op("||", alias = true)
  def or(x: A, y: A): Boolean
  @op("^", alias = true)
  def xor(x: A, y: A): Boolean
  @op("unary_!", alias = true)
  def not(x: A): Boolean
}

@typeclass trait Textual[S] extends TensorType[S]

case object TensorTypeFloat extends Floating[Float] {
  override def tag: DataType = TensorType.FloatTag
  override def jtag: Class[_ <: TType] = classOf[TFloat32]
  override def classTag: ClassTag[Float] = scala.reflect.classTag[Float]
  override def codec: TensorCodec[Float] = FloatTensorCodec
  override def one: Float = 1.0f
  override def zero: Float = 0.0f
  override def plus[B](left: Float, right: B)(implicit c: Convertible[B, Float]): Float =
    left + Convertible[B, Float].convert(right)
  override def multiply[B](left: Float, right: B)(implicit c: Convertible[B, Float]): Float =
    left * Convertible[B, Float].convert(right)
  override def minus[B](left: Float, right: B)(implicit c: Convertible[B, Float]): Float =
    left - Convertible[B, Float].convert(right)
  override def negate(a: Float): Float = -a
  override def div[B](left: Float, right: B)(implicit c: Convertible[B, Float]): Float =
    left / Convertible[B, Float].convert(right)
  override def compare(x: Float, y: Float): Int = x.compareTo(y)
}

case object TensorTypeDouble extends Floating[Double] {
  override def tag: DataType = TensorType.DoubleTag
  override def jtag: Class[_ <: TType] = classOf[TFloat64]
  override def classTag: ClassTag[Double] = scala.reflect.classTag[Double]
  override def codec: TensorCodec[Double] = DoubleTensorCodec
  override def one: Double = 1.0d
  override def zero: Double = 0.0d
  override def plus[B](left: Double, right: B)(implicit c: Convertible[B, Double]): Double =
    left + Convertible[B, Double].convert(right)
  override def multiply[B](left: Double, right: B)(implicit c: Convertible[B, Double]): Double =
    left * Convertible[B, Double].convert(right)
  override def minus[B](left: Double, right: B)(implicit c: Convertible[B, Double]): Double =
    left - Convertible[B, Double].convert(right)
  override def negate(a: Double): Double = -a
  override def div[B](left: Double, right: B)(implicit c: Convertible[B, Double]): Double =
    left / Convertible[B, Double].convert(right)
  override def compare(x: Double, y: Double): Int = x.compareTo(y)
}

case object TensorTypeLong extends Floating[Long] {
  override def tag: DataType = TensorType.LongTag
  override def jtag: Class[_ <: TType] = classOf[TInt64]
  override def classTag: ClassTag[Long] = scala.reflect.classTag[Long]
  override def codec: TensorCodec[Long] = LongTensorCodec
  override def one: Long = 1L
  override def zero: Long = 0L
  override def plus[B](left: Long, right: B)(implicit c: Convertible[B, Long]): Long =
    left + Convertible[B, Long].convert(right)
  override def multiply[B](left: Long, right: B)(implicit c: Convertible[B, Long]): Long =
    left * Convertible[B, Long].convert(right)
  override def minus[B](left: Long, right: B)(implicit c: Convertible[B, Long]): Long =
    left - Convertible[B, Long].convert(right)
  override def negate(a: Long): Long = -a
  override def div[B](left: Long, right: B)(implicit c: Convertible[B, Long]): Long =
    left / Convertible[B, Long].convert(right)
  override def compare(x: Long, y: Long): Int = x.compareTo(y)
}

case object TensorTypeInt extends Numeric[Int] {
  override def tag: DataType = TensorType.IntTag
  override def jtag: Class[_ <: TType] = classOf[TInt32]
  override def classTag: ClassTag[Int] = scala.reflect.classTag[Int]
  override def codec: TensorCodec[Int] = IntTensorCodec
  override def one: Int = 1
  override def zero: Int = 0
  override def plus[B](left: Int, right: B)(implicit c: Convertible[B, Int]): Int =
    left + Convertible[B, Int].convert(right)
  override def multiply[B](left: Int, right: B)(implicit c: Convertible[B, Int]): Int =
    left * Convertible[B, Int].convert(right)
  override def minus[B](left: Int, right: B)(implicit c: Convertible[B, Int]): Int =
    left - Convertible[B, Int].convert(right)
  override def negate(a: Int): Int = -a
  override def div[B](left: Int, right: B)(implicit c: Convertible[B, Int]): Int =
    left / Convertible[B, Int].convert(right)
  override def compare(x: Int, y: Int): Int = x.compareTo(y)
}

case object TensorTypeByte extends Numeric[Byte] {
  override def tag: DataType = TensorType.ByteTag
  override def jtag: Class[_ <: TType] = classOf[TUint8]
  override def classTag: ClassTag[Byte] = scala.reflect.classTag[Byte]
  override def codec: TensorCodec[Byte] = ByteTensorCodec
  override def one: Byte = 1
  override def zero: Byte = 0
  override def plus[B](left: Byte, right: B)(implicit c: Convertible[B, Byte]): Byte =
    (left + Convertible[B, Byte].convert(right)).toByte
  override def multiply[B](left: Byte, right: B)(implicit c: Convertible[B, Byte]): Byte =
    (left * Convertible[B, Byte].convert(right)).toByte
  override def minus[B](left: Byte, right: B)(implicit c: Convertible[B, Byte]): Byte =
    (left - Convertible[B, Byte].convert(right)).toByte
  override def negate(a: Byte): Byte = (-a).toByte
  override def div[B](left: Byte, right: B)(implicit c: Convertible[B, Byte]): Byte =
    (left / Convertible[B, Byte].convert(right)).toByte
  override def compare(x: Byte, y: Byte): Int = x.compareTo(y)
}

case object TensorTypeBoolean extends Logical[Boolean] {
  override def tag: DataType = TensorType.BoolTag
  override def jtag: Class[_ <: TType] = classOf[TBool]
  override def classTag: ClassTag[Boolean] = scala.reflect.classTag[Boolean]
  override def codec: TensorCodec[Boolean] = BooleanTensorCodec
  override def and(x: Boolean, y: Boolean): Boolean = x && y
  override def or(x: Boolean, y: Boolean): Boolean = x || y
  override def xor(x: Boolean, y: Boolean): Boolean = x ^ y
  override def not(x: Boolean): Boolean = !x
}

case object TensorTypeString extends Textual[String] {
  override def tag: DataType = TensorType.StringType
  override def jtag: Class[_ <: TType] = classOf[TString]
  override def classTag: ClassTag[String] = scala.reflect.classTag[String]
  override def codec: TensorCodec[String] = StringTensorCodec
}

object TensorType {

  val FloatTag: DataType = DataType.DT_FLOAT
  val DoubleTag: DataType = DataType.DT_DOUBLE
  val LongTag: DataType = DataType.DT_INT64
  val IntTag: DataType = DataType.DT_INT32
  val ByteTag: DataType = DataType.DT_UINT8
  val BoolTag: DataType = DataType.DT_BOOL
  val StringType: DataType = DataType.DT_STRING

  // reverse lookup, might be used for deserialization
  def of(dataType: DataType): TensorType[_] = {
    import syntax._
    dataType match {
      case FloatTag   => TensorType[Float]
      case DoubleTag  => TensorType[Double]
      case LongTag    => TensorType[Long]
      case IntTag     => TensorType[Int]
      case ByteTag    => TensorType[Byte]
      case BoolTag    => TensorType[Boolean]
      case StringType => TensorType[String]
      case _          => throw new IllegalArgumentException(s"data type $dataType is not supported")
    }
  }

  def of(code: Int): TensorType[_] = {
    val dataType = DataType.values().find(t => t.getNumber == code).get
    of(dataType)
  }

  trait Instances {
    implicit val floatTfTypeInst: Floating[Float] = TensorTypeFloat
    implicit val doubleTfTypeInst: Floating[Double] = TensorTypeDouble
    implicit val longTfTypeInst: Numeric[Long] = TensorTypeLong
    implicit val intTfTypeInst: Numeric[Int] = TensorTypeInt
    implicit val byteTfTypeInst: Numeric[Byte] = TensorTypeByte
    implicit val booleanTfTypeInst: Logical[Boolean] = TensorTypeBoolean
    implicit val stringTfTypeInst: Textual[String] = TensorTypeString
  }

  trait AllSyntax
      extends TensorType.Instances
      with TensorType.ToTensorTypeOps
      with Order.ToOrderOps
      with Eq.ToEqOps
      with Semiring.ToSemiringOps
      with Rng.ToRngOps
      with Rig.ToRigOps
      with Ring.ToRingOps
      with Field.ToFieldOps
      with Numeric.ToNumericOps
      with Logical.ToLogicalOps
      with Textual.ToTextualOps

  object syntax extends AllSyntax
}
