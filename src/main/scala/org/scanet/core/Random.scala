package org.scanet.core

import simulacrum.typeclass

import scala.annotation.tailrec
import scala.reflect.ClassTag

class Random[A: Dist](val gen: Generator) {

  def next: (Random[A], A) = {
    val (nextGen, value) = Dist[A].generate(gen)
    (new Random[A](nextGen), value)
  }

  def next[B >: A : ClassTag: Dist](n: Int): (Random[B], Array[B]) = {
    val dist = Dist[B]
    @tailrec
    def fill(gen: Generator, acc: Array[B], index: Int): (Generator, Array[B], Int) = {
      if (index == acc.length) {
        (gen, acc, index)
      } else {
        val (nextGen, value) = dist.generate(gen)
        acc(index) = value
        fill(nextGen, acc, index + 1)
      }
    }
    val (nextGen, acc, _) = fill(gen, Array.ofDim(n), 0)
    (Random[B](nextGen), acc)
  }
}

object Random {
  def apply[A: Dist](gen: Generator): Random[A] = new Random(gen)
}

class Generator(val seed: Long, val next: Long => Long) {
  def generate: (Generator, Long) = {
    val value = next(seed)
    (new Generator(value, next), value)
  }
}

object Generator {

  def uniform: Generator = uniform(new java.util.Random().nextLong())

  /**
   * Uniform generator can generate a stream of pseudorandom numbers.
   * The class uses a 48-bit seed, which is modified using a linear congruential formula.
   * (See Donald Knuth, The Art of Computer Programming, Volume 2, Section 3.2.1.)
   *
   * @param seed for pseudorandom generators to start
   * @return generator with next seed
   */
  def uniform(seed: Long): Generator = {
    val multiplier: Long = 0x5DEECE66DL
    val addend: Long = 0xBL
    val mask: Long = (1L << 48) - 1
    new Generator(seed, prevSeed => {
      (prevSeed * multiplier + addend) & mask
    })
  }
}

@typeclass trait Dist[A] {
  def generate(from: Generator): (Generator, A)
}

object Dist {

  trait Instances {
    implicit def distDouble: Dist[Double] = new DistDouble {}
    implicit def distFloat: Dist[Float] = new DistFloat {}
    implicit def distLong: Dist[Long] = new DistLong {}
    implicit def distInt: Dist[Int] = new DistInt {}
    implicit def distShort: Dist[Short] = new DistShort {}
    implicit def distByte: Dist[Byte] = new DistByte {}
  }

  trait Syntax extends Instances with Dist.ToDistOps

  object syntax extends Syntax
}

trait DistDouble extends Dist[Double] {
  override def generate(gen: Generator): (Generator, Double) = {
    val (nextGen1, value1) = gen.generate
    val (nextGen2, value2) = nextGen1.generate
    (nextGen2, (((value1 >>> 22).longValue() << 27).longValue()
      + (value2 >>> 21).longValue()).toDouble / (1L << 53))
  }
}

trait DistFloat extends Dist[Float] {
  override def generate(gen: Generator): (Generator, Float) = {
    val (nextGen, value) = gen.generate
    (nextGen, (value >>> 24).toInt / (1 << 24).toFloat)
  }
}

trait DistLong extends Dist[Long] {
  override def generate(gen: Generator): (Generator, Long) = {
    val (nextGen1, value1) = gen.generate
    val (nextGen2, value2) = nextGen1.generate
    (nextGen2, ((value1 >>> 16).longValue() << 32).longValue() + (value2 >>> 16).longValue())
  }
}

trait DistInt extends Dist[Int] {
  override def generate(gen: Generator): (Generator, Int) = {
    val (nextGen, value) = gen.generate
    (nextGen, (value >>> 16).toInt)
  }
}

trait DistShort extends Dist[Short] {
  override def generate(gen: Generator): (Generator, Short) = {
    val (nextGen, value) = gen.generate
    (nextGen, (value >>> 16).toShort)
  }
}

trait DistByte extends Dist[Byte] {
  override def generate(gen: Generator): (Generator, Byte) = {
    val (nextGen, value) = gen.generate
    (nextGen, (value >>> 16).toByte)
  }
}