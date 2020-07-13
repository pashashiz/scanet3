package org.scanet.core

trait CoreSyntax extends TensorType.Syntax with Slice.Syntax
  with Session.Syntax with InstantEvalOps.Syntax with ConstOp.Syntax with CoreOp.Syntax {
}
