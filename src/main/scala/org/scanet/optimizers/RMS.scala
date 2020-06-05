package org.scanet.optimizers

import org.scanet.core.Output
import org.scanet.math.syntax._

trait RMS {

  def rho: Float

  // root mean squared
  def rms(x: Output[Float]): Output[Float] = {
    (x + 1e-7f.const).sqrt
  }

  // running (decaying) average
  def avg(prev: Output[Float], curr: Output[Float]): Output[Float] = {
    rho.const * prev + (1 - rho).const * curr.sqr
  }
}
