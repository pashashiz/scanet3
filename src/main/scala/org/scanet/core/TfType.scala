package org.scanet.core

import org.tensorflow.DataType
import simulacrum.typeclass

import scala.reflect.ClassTag

@typeclass trait TfType[A] {
  def tag: DataType
  def classTag: ClassTag[A]
  def show: String = classTag.toString()
}

trait TfTypeFloat extends TfType[Float] {
  override def tag: DataType = TfType.FloatTag
  override def classTag: ClassTag[Float] = scala.reflect.classTag[Float]
}

trait TfTypeDouble extends TfType[Double] {
  override def tag: DataType = TfType.DoubleTag
  override def classTag: ClassTag[Double] = scala.reflect.classTag[Double]
}

trait TfTypeLong extends TfType[Long] {
  override def tag: DataType = TfType.LongTag
  override def classTag: ClassTag[Long] = scala.reflect.classTag[Long]
}

trait TfTypeInt extends TfType[Int] {
  override def tag: DataType = TfType.IntTag
  override def classTag: ClassTag[Int] = scala.reflect.classTag[Int]
}

trait TfTypeByte extends TfType[Byte] {
  override def tag: DataType = TfType.ByteTag
  override def classTag: ClassTag[Byte] = scala.reflect.classTag[Byte]
}

trait TfTypeString extends TfType[String] {
  override def tag: DataType = TfType.StringType
  override def classTag: ClassTag[String] = scala.reflect.classTag[String]
}

object TfType {

  val FloatTag: DataType = DataType.FLOAT
  val DoubleTag: DataType = DataType.DOUBLE
  val LongTag: DataType = DataType.INT64
  val IntTag: DataType = DataType.INT32
  val ByteTag: DataType = DataType.UINT8
  val StringType: DataType = DataType.STRING

  trait Instances {
    implicit def floatTfTypeInst: TfType[Float] = new TfTypeFloat {}
    implicit def doubleTfTypeInst: TfType[Double] = new TfTypeDouble {}
    implicit def longTfTypeInst: TfType[Long] = new TfTypeLong {}
    implicit def intTfTypeInst: TfType[Int] = new TfTypeInt {}
    implicit def byteTfTypeInst: TfType[Byte] = new TfTypeByte {}
    implicit def stringTfTypeInst: TfType[String] = new TfTypeString {}
  }
  trait Syntax extends TfType.Instances with TfType.ToTfTypeOps
  object syntax extends Syntax
}
