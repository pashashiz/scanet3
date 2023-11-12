package scanet.models.layer

import scanet.core.{Expr, Floating, Params, Path, Shape}
import scanet.math.syntax._
import scanet.models.ParamDef

import scala.collection.immutable.Seq

/** Layer which composes multiple other layers sequentially
  *
  * @param layers layers to compose from left to right, such as
  *               {{{layer 1 >>> layer 2 >>> ... >>> layer last}}}
  */
case class Composed private (layers: Seq[Layer]) extends Layer {

  override def stateful: Boolean = layers.forall(_.stateful)

  override def params(input: Shape): Params[ParamDef] = {
    val (_, layerParams) = layers.zipWithIndex
      .foldLeft((input, Seq.empty[Params[ParamDef]])) {
        case ((in, paramsAcc), (layer, index)) =>
          val out = layer.outputShape(in)
          val params = layer.params(in).prependPath(index)
          (out, params +: paramsAcc)
      }
    layerParams.reduce(_ ++ _)
  }

  private def recoverLayerParams[E: Floating](params: Params[Expr[E]]): Map[Int, Params[Expr[E]]] =
    params.unwrap
      .toList
      .map {
        case (path, expr) =>
          (path.segments.head.toInt, (Path(path.segments.tail), expr))
      }
      .groupBy(_._1)
      .map {
        case (index, values) =>
          (index, Params(values.map { case (_, kv) => kv }: _*))
      }

  override def build[E: Floating](
      input: Expr[E],
      params: Params[Expr[E]]): (Expr[E], Params[Expr[E]]) = {
    val layerParams = recoverLayerParams(params)
    val (out, stateParams) = layers.zipWithIndex
      .foldLeft((input, Seq.empty[Params[Expr[E]]])) {
        case ((in, stateParamsAcc), (layer, index)) =>
          val (out, stateParams) = layer.build(in, layerParams.getOrElse(index, Params.empty))
          (out, stateParams.prependPath(index) +: stateParamsAcc)
      }
    (out, stateParams.reduce(_ ++ _))
  }

  override def penalty[E: Floating](params: Params[Expr[E]]): Expr[E] = {
    val layerParams = recoverLayerParams(params)
    val layerPenalty = layers.zipWithIndex
      .foldLeft(Seq.empty[Expr[E]]) {
        case (penaltyAcc, (layer, index)) =>
          val penalty = layer.penalty(layerParams.getOrElse(index, Params.empty))
          penalty +: penaltyAcc
      }
    plus(layerPenalty)
  }

  override def outputShape(input: Shape): Shape =
    layers.foldLeft(input) {
      case (in, layer) => layer.outputShape(in)
    }

  override def info(input: Shape): Seq[LayerInfo] = {
    val (_, layersInfo) = layers.foldLeft((input, Seq.empty[LayerInfo])) {
      case ((in, acc), layer) =>
        (layer.outputShape(in), acc ++ layer.info(in))
    }
    layersInfo
  }

  override def makeTrainable(trainable: Boolean): Layer =
    copy(layers = layers.map(_.makeTrainable(trainable)))

  override def trainable: Boolean = layers.exists(_.trainable)

  override def toString: String = layers.mkString(" >> ")
}

object Composed {

  def apply(layers: Seq[Layer]): Layer = {
    require(layers.nonEmpty, "at least 1 layer is required")
    val flatten = layers.flatMap {
      case Composed(nested) => nested
      case regular          => Seq(regular)
    }
    flatten match {
      case single +: Nil => single
      case multiple      => new Composed(multiple)
    }
  }

  def apply(first: Layer, rest: Layer*): Layer =
    Composed((first +: rest).toList)
}
