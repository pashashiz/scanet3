package org.scanet.optimizers

import org.scanet.core.{Output, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

abstract class TensorFunctionBuilder[A1: Numeric: TensorType, A2: Numeric: TensorType, B: Numeric: TensorType] extends (Output[A1] => TensorFunction[A2, B])
