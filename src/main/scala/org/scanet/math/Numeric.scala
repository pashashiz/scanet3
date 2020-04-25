package org.scanet.math

import simulacrum.{op, typeclass}

@typeclass trait Eq[A] {
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

@typeclass trait Semiring[A] {
  // todo: figure out why + operator is not resolved
  @op("+", alias = true)
  def plus[B](left: A, right: B)(implicit c: Convertible[B, A]): A
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