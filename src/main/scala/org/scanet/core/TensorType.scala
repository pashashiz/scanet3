package org.scanet.core

import org.tensorflow.proto.framework.DataType
import org.tensorflow.types._
import org.tensorflow.types.family.TType
import simulacrum.typeclass

import scala.reflect.ClassTag

@typeclass sealed trait TensorType[A] {
  def tag: DataType
  def jtag: Class[_ <: TType]
  def code: Int = tag.getNumber
  def classTag: ClassTag[A]
  def show: String = classTag.toString()
  def codec: TensorCodec[A]
}

case object TensorTypeFloat extends TensorType[Float] {
  override def tag: DataType = TensorType.FloatTag
  override def jtag: Class[_ <: TType] = classOf[TFloat32]
  override def classTag: ClassTag[Float] = scala.reflect.classTag[Float]
  override def codec: TensorCodec[Float] = FloatTensorCodec
}

case object TensorTypeDouble extends TensorType[Double] {
  override def tag: DataType = TensorType.DoubleTag
  override def jtag: Class[_ <: TType] = classOf[TFloat64]
  override def classTag: ClassTag[Double] = scala.reflect.classTag[Double]
  override def codec: TensorCodec[Double] = DoubleTensorCodec
}

case object TensorTypeLong extends TensorType[Long] {
  override def tag: DataType = TensorType.LongTag
  override def jtag: Class[_ <: TType] = classOf[TInt64]
  override def classTag: ClassTag[Long] = scala.reflect.classTag[Long]
  override def codec: TensorCodec[Long] = LongTensorCodec
}

case object TensorTypeInt extends TensorType[Int] {
  override def tag: DataType = TensorType.IntTag
  override def jtag: Class[_ <: TType] = classOf[TInt32]
  override def classTag: ClassTag[Int] = scala.reflect.classTag[Int]
  override def codec: TensorCodec[Int] = IntTensorCodec
}

case object TensorTypeByte extends TensorType[Byte] {
  override def tag: DataType = TensorType.ByteTag
  override def jtag: Class[_ <: TType] = classOf[TUint8]
  override def classTag: ClassTag[Byte] = scala.reflect.classTag[Byte]
  override def codec: TensorCodec[Byte] = ByteTensorCodec
}

case object TensorTypeBoolean extends TensorType[Boolean] {
  override def tag: DataType = TensorType.BoolTag
  override def jtag: Class[_ <: TType] = classOf[TBool]
  override def classTag: ClassTag[Boolean] = scala.reflect.classTag[Boolean]
  override def codec: TensorCodec[Boolean] = BooleanTensorCodec
}

case object TensorTypeString extends TensorType[String] {
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

    implicit val floatTfTypeInst: TensorType[Float] = TensorTypeFloat
    implicit val doubleTfTypeInst: TensorType[Double] = TensorTypeDouble
    implicit val longTfTypeInst: TensorType[Long] = TensorTypeLong
    implicit val intTfTypeInst: TensorType[Int] = TensorTypeInt
    implicit val byteTfTypeInst: TensorType[Byte] = TensorTypeByte
    implicit val booleanTfTypeInst: TensorType[Boolean] = TensorTypeBoolean
    implicit val stringTfTypeInst: TensorType[String] = TensorTypeString
  }
  trait Syntax extends TensorType.Instances with TensorType.ToTensorTypeOps
  object syntax extends Syntax
}
