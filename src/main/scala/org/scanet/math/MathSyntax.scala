package org.scanet.math

import org.scanet.core.CoreSyntax

trait MathSyntax
    extends CoreSyntax
    with Dist.Syntax
    with Numeric.Syntax
    with Logical.Syntax
    with alg.kernels.Syntax
    with logical.kernels.Syntax
    with grad.Syntax
