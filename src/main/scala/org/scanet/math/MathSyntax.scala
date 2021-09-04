package org.scanet.math

import org.scanet.core.CoreSyntax

trait MathSyntax
    extends CoreSyntax
    with Dist.AllSyntax
    with Numeric.AllSyntax
    with Logical.AllSyntax
    with alg.kernels.AllSyntax
    with logical.kernels.AllSyntax
    with grad.GradSyntax
