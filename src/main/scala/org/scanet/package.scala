package org

import org.scanet.core.CoreSyntax
import org.scanet.math.MathSyntax
import org.scanet.strings.StringSyntax

package object scanet {

  object syntax extends CoreSyntax with MathSyntax with StringSyntax

}
