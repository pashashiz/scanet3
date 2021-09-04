package org.scanet.strings

import org.scanet.core.CoreSyntax

trait StringsSyntax extends CoreSyntax with Textual.AllSyntax with kernels.AllSyntax
object syntax extends StringsSyntax
