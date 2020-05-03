package org.scanet.math

import org.scanet.core.CoreSyntax

trait MathSyntax extends CoreSyntax with Dist.Syntax
  with NumericPrimitives.Syntax with Logical.Syntax
  with MathBaseOp.Syntax with MathLogicalOp.Syntax
  with MathGradOp.Syntax
