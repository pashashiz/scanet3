package org.scanet

import org.scanet.core.{Eval, Slice}
import org.scanet.math.{Convertable, Dist, Numeric}

package object syntax {

  trait MathSyntax extends Convertable.Syntax with Dist.Syntax with Numeric.Syntax

  object math extends MathSyntax

  // todo: remove MathSyntax when Numeric is removed from tensor
  trait CoreSyntax extends Slice.Syntax with Eval.Syntax with MathSyntax

  object core extends CoreSyntax

}
