package org.scanet.math

import org.scanet.core.CoreSyntax

trait MathSyntax extends CoreSyntax with Dist.Syntax with NumericPrimitives.Syntax with MathOp.Syntax
