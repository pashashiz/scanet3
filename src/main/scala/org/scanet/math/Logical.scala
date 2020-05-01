package org.scanet.math

import simulacrum.{op, typeclass}

@typeclass trait Logical[A] {
  @op("&&", alias = true)
  def and(x: A, y: A): Boolean
  @op("||", alias = true)
  def or(x: A, y: A): Boolean
  @op("^", alias = true)
  def xor(x: A, y: A): Boolean
  @op("unary_!", alias = true)
  def not(x: A): Boolean
}

object Logical {
  trait Instances {
    implicit def booleanIsLogical: Logical[Boolean] = new BooleanIsLogical
  }
  trait Syntax extends Instances with Logical.ToLogicalOps
  object syntax extends Syntax
}

class BooleanIsLogical extends Logical[Boolean] {
  override def and(x: Boolean, y: Boolean): Boolean = x && y
  override def or(x: Boolean, y: Boolean): Boolean = x || y
  override def xor(x: Boolean, y: Boolean): Boolean = x ^ y
  override def not(x: Boolean): Boolean = !x
}