package org.scanet.math

import org.scanet.core.CoreSyntax

trait MathSyntax extends CoreSyntax with Dist.Syntax
  with Numeric.Syntax with Logical.Syntax
  with MathBaseOp.Syntax with MathExtraOp.Syntax with MathLogicalOp.Syntax
  with MathGradOp.Syntax
