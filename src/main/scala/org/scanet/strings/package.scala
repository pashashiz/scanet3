package org.scanet

import org.scanet.core.CoreSyntax

package object strings {
  trait StringSyntax extends CoreSyntax with Textual.Syntax with kernels.Syntax
  object syntax extends StringSyntax
}
