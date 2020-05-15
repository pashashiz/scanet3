package org.scanet.optimizers

import org.scanet.core.Output

trait Algorithm {

  def delta(grad: Output[Float]): Output[Float]
}