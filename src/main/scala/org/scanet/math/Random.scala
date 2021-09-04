package org.scanet.math

import simulacrum.typeclass

import scala.annotation.tailrec
import scala.reflect.ClassTag

class Random[A: Dist](val gen: Generator, range: Option[(A, A)]) {

  def next: (Random[A], A) = {
    val (nextGen, value) = Dist[A].generate(gen, range)
    (new Random[A](nextGen, range), value)
  }

  def next[B >: A : ClassTag: Dist](n: Int): (Random[B], Array[B]) = {
    val dist = Dist[B]
    @tailrec
    def fill(gen: Generator, acc: Array[B], index: Int): (Generator, Array[B], Int) = {
      if (index == acc.length) {
        (gen, acc, index)
      } else {
        val (nextGen, value) = dist.generate(gen, range)
        acc(index) = value
        fill(nextGen, acc, index + 1)
      }
    }
    val (nextGen, acc, _) = fill(gen, Array.ofDim(n), 0)
    (Random[B](nextGen), acc)
  }
}

object Random {
  def apply[A: Dist](gen: Generator, range: Option[(A, A)] = None): Random[A] = new Random(gen, range)
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
  def generate(from: Generator, range: Option[(A, A)]): (Generator, A)
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

  trait AllSyntax extends Instances with Dist.ToDistOps

  object syntax extends AllSyntax
}

trait DistDouble extends Dist[Double] {

  override def generate(gen: Generator, range: Option[(Double, Double)]): (Generator, Double) = {
    val (nextGen1, seed1) = gen.generate
    val (nextGen2, seed2) = nextGen1.generate
    val value = (((seed1 >>> 22).longValue() << 27).longValue()
      + (seed2 >>> 21).longValue()).toDouble / (1L << 53)
    val scaled = range match {
      case Some((from, until)) =>
        val scale = until - from
        val shift = from - 0
        value * scale + shift
      case None => value
    }
    (nextGen2, scaled)
  }
}

trait DistFloat extends Dist[Float] {
  override def generate(gen: Generator, range: Option[(Float, Float)]): (Generator, Float) = {
    val (nextGen, seed) = gen.generate
    val value = (seed >>> 24).toInt / (1 << 24).toFloat
    val scaled = range match {
      case Some((from, until)) =>
        val scale = until - from
        val shift = from - 0
        value * scale + shift
      case None => value
    }
    (nextGen, scaled)
  }
}

trait DistLong extends Dist[Long] {
  override def generate(gen: Generator, range: Option[(Long, Long)]): (Generator, Long) = {
    val (nextGen1, seed1) = gen.generate
    val (nextGen2, seed2) = nextGen1.generate
    val value = ((seed1 >>> 16).longValue() << 32).longValue() + (seed2 >>> 16).longValue()
    val scaled = range match {
      case Some((from, until)) =>
        val scale = (until - from) / Long.MaxValue
        val shift = from - 0
        value * scale + shift
      case None => value
    }
    (nextGen2, scaled)
  }
}

trait DistInt extends Dist[Int] {
  override def generate(gen: Generator, range: Option[(Int, Int)]): (Generator, Int) = {
    val (nextGen, seed) = gen.generate
    val value = (seed >>> 16).toInt
    val scaled = range match {
      case Some((from, until)) =>
        val scale = (until - from) / Int.MaxValue
        val shift = from - 0
        value * scale + shift
      case None => value
    }
    (nextGen, scaled)
  }
}

trait DistShort extends Dist[Short] {
  override def generate(gen: Generator, range: Option[(Short, Short)]): (Generator, Short) = {
    val (nextGen, seed) = gen.generate
    val value = (seed >>> 16).toShort
    val scaled = range match {
      case Some((from, until)) =>
        val scale = (until - from) / Short.MaxValue
        val shift = from - 0
        (value * scale + shift).toShort
      case None => value
    }
    (nextGen, scaled)
  }
}

trait DistByte extends Dist[Byte] {
  override def generate(gen: Generator, range: Option[(Byte, Byte)]): (Generator, Byte) = {
    val (nextGen, seed) = gen.generate
    val value = (seed >>> 16).toByte
    val scaled = range match {
      case Some((from, until)) =>
        val scale = (until - from) / Byte.MaxValue
        val shift = from - 0
        (value * scale + shift).toByte
      case None => value
    }
    (nextGen, scaled)
  }
}