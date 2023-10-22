package scanet.core

case class ShowExpr(expr: Expr[_]) {

  private val outerRefCount: Map[String, Int] = {
    def scan(
        parent: String,
        expr: Expr[_],
        refs: Map[String, Set[String]]): Map[String, Set[String]] = {
      refs.get(expr.ref) match {
        case Some(existing) =>
          refs + (expr.ref -> (existing + parent))
        case None =>
          val nextRefs = (expr.inputs ++ expr.controls)
            .foldLeft(refs) {
              case (nextRefs, inner) =>
                scan(expr.ref, inner, nextRefs)
            }
          nextRefs + (expr.ref -> Set(parent))
      }
    }
    scan("root", expr, Map.empty[String, Set[String]])
      .map { case (k, v) => (k, v.size) }
  }

  private def showRec(expr: Expr[_], refs: Map[String, String]): (String, Map[String, String]) = {
    if (refs.contains(expr.ref)) {
      (expr.ref, refs)
    } else {
      val (inputs, refs2) = showRecN(expr.inputs, refs)
      val (controls, refs3) = showRecN(expr.controls, refs2)
      val value = expr.value.fold("")(v => s"($v)")
      val args = if (inputs.nonEmpty) inputs.mkString("(", ",", ")") else ""
      val deps = if (controls.nonEmpty) ".depends" + controls.mkString("(", ",", ")") else ""
      val fullName =
        if (expr.label == expr.name) s"${expr.name}"
        else s"${expr.label}:${expr.name}"
      val tpeOrEmpty = expr.tpe.map(t => s"[${t.show}]").getOrElse("")
      (s"$fullName$value$args$deps$tpeOrEmpty:${expr.shape}", refs3)
    }
  }

  private def showRecN(
      exprs: Seq[Expr[_]],
      refs: Map[String, String]): (Seq[String], Map[String, String]) =
    exprs.foldLeft((Seq.empty[String], refs)) {
      case ((repr, accRefs), exprChild) =>
        val (childRepr, childRefs) = showRec(exprChild, accRefs)
        if (outerRefCount(exprChild.ref) == 1) {
          (repr :+ childRepr, accRefs ++ childRefs)
        } else {
          val accRefs2 = accRefs ++ childRefs
          val accRefs3 =
            if (accRefs2.contains(exprChild.ref)) accRefs2
            else accRefs2 + (exprChild.ref -> childRepr)
          (repr :+ s"#${exprChild.ref}", accRefs3)
        }
    }

  def show: String = {
    val (repr, refs) = showRec(expr, Map.empty)
    (repr +: refs.toSeq.map { case (ref, repr) => s"$ref: $repr" }).mkString("\n")
  }
}
