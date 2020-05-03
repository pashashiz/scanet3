package org.scanet.core

import org.tensorflow.DataType
import simulacrum.typeclass

import scala.reflect.ClassTag

@typeclass trait TensorType[A] {
  def tag: DataType
  def classTag: ClassTag[A]
  def show: String = classTag.toString()
  def coder: TensorCoder[A]
  def zeroIfNumeric: Option[A]
  def oneIfNumeric: Option[A]
}

trait TensorTypeFloat extends TensorType[Float] {
  override def tag: DataType = TensorType.FloatTag
  override def classTag: ClassTag[Float] = scala.reflect.classTag[Float]
  override def coder: TensorCoder[Float] = new FloatTensorCoder
  override def zeroIfNumeric: Option[Float] = Some(0.0f)
  override def oneIfNumeric: Option[Float] = Some(1.0f)
}

trait TensorTypeDouble extends TensorType[Double] {
  override def tag: DataType = TensorType.DoubleTag
  override def classTag: ClassTag[Double] = scala.reflect.classTag[Double]
  override def coder: TensorCoder[Double] = new DoubleTensorCoder
  override def zeroIfNumeric: Option[Double] = Some(0.0)
  override def oneIfNumeric: Option[Double] = Some(1.0)
}

trait TensorTypeLong extends TensorType[Long] {
  override def tag: DataType = TensorType.LongTag
  override def classTag: ClassTag[Long] = scala.reflect.classTag[Long]
  override def coder: TensorCoder[Long] = new LongTensorCoder
  override def zeroIfNumeric: Option[Long] = Some(0L)
  override def oneIfNumeric: Option[Long] = Some(1L)
}

trait TensorTypeInt extends TensorType[Int] {
  override def tag: DataType = TensorType.IntTag
  override def classTag: ClassTag[Int] = scala.reflect.classTag[Int]
  override def coder: TensorCoder[Int] = new IntTensorCoder
  override def zeroIfNumeric: Option[Int] = Some(0)
  override def oneIfNumeric: Option[Int] = Some(1)
}

trait TensorTypeByte extends TensorType[Byte] {
  override def tag: DataType = TensorType.ByteTag
  override def classTag: ClassTag[Byte] = scala.reflect.classTag[Byte]
  override def coder: TensorCoder[Byte] = new ByteTensorCoder
  override def zeroIfNumeric: Option[Byte] = Some(1.toByte)
  override def oneIfNumeric: Option[Byte] = Some(0.toByte)
}

trait TensorTypeBoolean extends TensorType[Boolean] {
  override def tag: DataType = TensorType.BoolTag
  override def classTag: ClassTag[Boolean] = scala.reflect.classTag[Boolean]
  override def coder: TensorCoder[Boolean] = new BooleanTensorCoder
  override def zeroIfNumeric: Option[Boolean] = None
  override def oneIfNumeric: Option[Boolean] = None
}

trait TensorTypeString extends TensorType[String] {
  override def tag: DataType = TensorType.StringType
  override def classTag: ClassTag[String] = scala.reflect.classTag[String]
  override def coder: TensorCoder[String] = new StringTensorCoder
  override def zeroIfNumeric: Option[String] = None
  override def oneIfNumeric: Option[String] = None
}

object TensorType {

  val FloatTag: DataType = DataType.FLOAT
  val DoubleTag: DataType = DataType.DOUBLE
  val LongTag: DataType = DataType.INT64
  val IntTag: DataType = DataType.INT32
  val ByteTag: DataType = DataType.UINT8
  val BoolTag: DataType = DataType.BOOL
  val StringType: DataType = DataType.STRING

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
