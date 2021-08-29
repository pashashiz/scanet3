package org.scanet.core

trait CoreSyntax
    extends TensorType.Syntax
    with Slice.Syntax
    with Session.Syntax
    with Eval.Syntax
    with Const.Syntax
    with kernels.Syntax {}

// import org.scanet.core.kernels._ all core kernels (functions)
// import org.scanet.core.kernels.syntax._ all core kernels (functions) + syntax
// import org.scanet.core.syntax._ all core syntax
