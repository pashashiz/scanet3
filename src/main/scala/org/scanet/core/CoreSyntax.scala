package org.scanet.core

trait CoreSyntax
    extends TensorType.AllSyntax
    with Slice.AllSyntax
    with Session.AllSyntax
    with Eval.AllSyntax
    with Const.AllSyntax
    with kernels.AllSyntax {}

// import org.scanet.core.kernels._ all core kernels (functions)
// import org.scanet.core.kernels.syntax._ all core kernels (functions) + syntax
// import org.scanet.core.syntax._ all core syntax
