package org.scanet.core

import org.scanet.native.RichDataType
import org.tensorflow.DataType
import simulacrum.typeclass

import scala.reflect.ClassTag

@typeclass trait TensorType[A] {
  def tag: DataType
  def code: Int = new RichDataType(tag).code
  def classTag: ClassTag[A]
  def show: String = classTag.toString()
  def coder: TensorCoder[A]
}

trait TensorTypeFloat extends TensorType[Float] {
  override def tag: DataType = TensorType.FloatTag
  override def classTag: ClassTag[Float] = scala.reflect.classTag[Float]
  override def coder: TensorCoder[Float] = new FloatTensorCoder
}

trait TensorTypeDouble extends TensorType[Double] {
  override def tag: DataType = TensorType.DoubleTag
  override def classTag: ClassTag[Double] = scala.reflect.classTag[Double]
  override def coder: TensorCoder[Double] = new DoubleTensorCoder
}

trait TensorTypeLong extends TensorType[Long] {
  override def tag: DataType = TensorType.LongTag
  override def classTag: ClassTag[Long] = scala.reflect.classTag[Long]
  override def coder: TensorCoder[Long] = new LongTensorCoder
}

trait TensorTypeInt extends TensorType[Int] {
  override def tag: DataType = TensorType.IntTag
  override def classTag: ClassTag[Int] = scala.reflect.classTag[Int]
  override def coder: TensorCoder[Int] = new IntTensorCoder
}

trait TensorTypeByte extends TensorType[Byte] {
  override def tag: DataType = TensorType.ByteTag
  override def classTag: ClassTag[Byte] = scala.reflect.classTag[Byte]
  override def coder: TensorCoder[Byte] = new ByteTensorCoder
}

trait TensorTypeBoolean extends TensorType[Boolean] {
  override def tag: DataType = TensorType.BoolTag
  override def classTag: ClassTag[Boolean] = scala.reflect.classTag[Boolean]
  override def coder: TensorCoder[Boolean] = new BooleanTensorCoder
}

trait TensorTypeString extends TensorType[String] {
  override def tag: DataType = TensorType.StringType
  override def classTag: ClassTag[String] = scala.reflect.classTag[String]
  override def coder: TensorCoder[String] = new StringTensorCoder
}

object TensorType {

  val FloatTag: DataType = DataType.FLOAT
  val DoubleTag: DataType = DataType.DOUBLE
  val LongTag: DataType = DataType.INT64
  val IntTag: DataType = DataType.INT32
  val ByteTag: DataType = DataType.UINT8
  val BoolTag: DataType = DataType.BOOL
  val StringType: DataType = DataType.STRING

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
      .find(t => new RichDataType(t).code == code)
      .get
    of(dataType)
  }

  implicit def toRich(dataType: DataType): RichDataType = new RichDataType(dataType)

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
