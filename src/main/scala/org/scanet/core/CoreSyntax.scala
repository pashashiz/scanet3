package org.scanet.core

trait CoreSyntax extends TensorType.Syntax with Slice.Syntax
  with Session.Syntax with Eval.Syntax with ConstOp.Syntax with CoreOp.Syntax
