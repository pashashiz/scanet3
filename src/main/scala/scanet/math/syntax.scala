package scanet.math

import scanet.core.CoreSyntax

trait MathSyntax
    extends CoreSyntax
    with Dist.AllSyntax
    with Numeric.AllSyntax
    with Logical.AllSyntax
    with alg.kernels.AllSyntax
    with linalg.kernels.AllSyntax
    with stat.kernels.AllSyntax
    with logical.kernels.AllSyntax
    with grad.GradSyntax

object syntax extends MathSyntax
