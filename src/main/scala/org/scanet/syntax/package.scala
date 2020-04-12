package org.scanet

import org.scanet.core.{Convertable, Dist, Numeric}
import org.scanet.linalg.Slice

package object syntax {

  trait CoreSyntax extends Convertable.Syntax with Dist.Syntax with Numeric.Syntax

  object core extends CoreSyntax

  trait LinalgSyntax extends CoreSyntax with Slice.Syntax

  object linalg extends LinalgSyntax
}
