package org.scanet.optimizers

import org.scanet.core.Output
import org.scanet.math.syntax._

trait RMS {

  def rho: Double

  def epsilon: Double

  // root mean squared
  def rms(x: Output[Float]): Output[Float] = {
    (x + epsilon.toFloat.const).sqrt
  }

  // running (decaying) average
  def avg(prev: Output[Float], curr: Output[Float]): Output[Float] = {
    rho.toFloat.const * prev + (1 - rho.toFloat).const * curr.sqr
  }
}
