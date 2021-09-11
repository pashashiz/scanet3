package scanet.core

trait CoreSyntax
    extends TensorType.AllSyntax
    with Slice.AllSyntax
    with Session.AllSyntax
    with Eval.AllSyntax
    with Const.AllSyntax
    with kernels.AllSyntax {}

object syntax extends CoreSyntax
