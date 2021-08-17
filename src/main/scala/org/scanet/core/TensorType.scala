package org.scanet.core

import org.tensorflow.proto.framework.DataType
import org.tensorflow.types._
import org.tensorflow.types.family.TType
import simulacrum.typeclass

import scala.reflect.ClassTag

@typeclass trait TensorType[A] {
  def tag: DataType
  def jtag: Class[_ <: TType]
  def code: Int = tag.getNumber
  def classTag: ClassTag[A]
  def show: String = classTag.toString()
  def coder: TensorCoder[A]
}

trait TensorTypeFloat extends TensorType[Float] {
  override def tag: DataType = TensorType.FloatTag
  override def jtag: Class[_ <: TType] = classOf[TFloat32]
  override def classTag: ClassTag[Float] = scala.reflect.classTag[Float]
  override def coder: TensorCoder[Float] = new FloatTensorCoder
}

trait TensorTypeDouble extends TensorType[Double] {
  override def tag: DataType = TensorType.DoubleTag
  override def jtag: Class[_ <: TType] = classOf[TFloat64]
  override def classTag: ClassTag[Double] = scala.reflect.classTag[Double]
  override def coder: TensorCoder[Double] = new DoubleTensorCoder
}

trait TensorTypeLong extends TensorType[Long] {
  override def tag: DataType = TensorType.LongTag
  override def jtag: Class[_ <: TType] = classOf[TInt64]
  override def classTag: ClassTag[Long] = scala.reflect.classTag[Long]
  override def coder: TensorCoder[Long] = new LongTensorCoder
}

trait TensorTypeInt extends TensorType[Int] {
  override def tag: DataType = TensorType.IntTag
  override def jtag: Class[_ <: TType] = classOf[TInt32]
  override def classTag: ClassTag[Int] = scala.reflect.classTag[Int]
  override def coder: TensorCoder[Int] = new IntTensorCoder
}

trait TensorTypeByte extends TensorType[Byte] {
  override def tag: DataType = TensorType.ByteTag
  override def jtag: Class[_ <: TType] = classOf[TUint8]
  override def classTag: ClassTag[Byte] = scala.reflect.classTag[Byte]
  override def coder: TensorCoder[Byte] = new ByteTensorCoder
}

trait TensorTypeBoolean extends TensorType[Boolean] {
  override def tag: DataType = TensorType.BoolTag
  override def jtag: Class[_ <: TType] = classOf[TBool]
  override def classTag: ClassTag[Boolean] = scala.reflect.classTag[Boolean]
  override def coder: TensorCoder[Boolean] = new BooleanTensorCoder
}

trait TensorTypeString extends TensorType[String] {
  override def tag: DataType = TensorType.StringType
  override def jtag: Class[_ <: TType] = classOf[TString]
  override def classTag: ClassTag[String] = scala.reflect.classTag[String]
  override def coder: TensorCoder[String] = new StringTensorCoder
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
      case FloatTag => TensorType[Float]
      case DoubleTag => TensorType[Double]
      case LongTag => TensorType[Long]
      case IntTag => TensorType[Int]
      case ByteTag => TensorType[Byte]
      case BoolTag => TensorType[Boolean]
      case StringType => TensorType[String]
      case _ => throw new IllegalArgumentException(s"data type $dataType is not supported")
    }
  }

  def of(code: Int): TensorType[_] = {
    val dataType = DataType.values()
      .find(t => t.getNumber == code)
      .get
    of(dataType)
  }

  trait Instances {

    implicit def floatTfTypeInst: TensorType[Float] = new TensorTypeFloat {}
    implicit def doubleTfTypeInst: TensorType[Double] = new TensorTypeDouble {}
    implicit def longTfTypeInst: TensorType[Long] = new TensorTypeLong {}
    implicit def intTfTypeInst: TensorType[Int] = new TensorTypeInt {}
    implicit def byteTfTypeInst: TensorType[Byte] = new TensorTypeByte {}
    implicit def booleanTfTypeInst: TensorType[Boolean] = new TensorTypeBoolean {}
    implicit def stringTfTypeInst: TensorType[String] = new TensorTypeString {}
  }
  trait Syntax extends TensorType.Instances with TensorType.ToTensorTypeOps
  object syntax extends Syntax
}
