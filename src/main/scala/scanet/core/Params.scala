package scanet.core

case class Path(segments: Seq[String]) {
  def /(child: Path): Path = Path(segments ++ child.segments)
  def startsWith(parent: Path): Boolean = segments.startsWith(parent.segments)
  def endsWith(parent: Path): Boolean = segments.endsWith(parent.segments)
  def relativeTo(parent: Path): Option[Path] =
    if (startsWith(parent)) Some(Path(segments.drop(parent.segments.size))) else None
  override def toString: String = segments.mkString("/")
}

object Path {
  def apply(s1: String, sn: String*): Path = new Path(s1 :: sn.toList)
  def parse(path: String): Path = new Path(path.split("/").toList)
  implicit def stringIsPath(path: String): Path = Path.parse(path)
  implicit def intIsPath(part: Int): Path = Path(part.toString)
}

// convenient wrapper over Map[Path, A] to get DSL like API
case class Params[A](params: Map[Path, A]) {
  def unwrap: Map[Path, A] = params
  def apply(path: Path): A =
    params.getOrElse(path, error(s"missing $path param"))
  def paths: Set[Path] = params.keySet
  def children(parent: Path): Params[A] = {
    val childrenParams = params.flatMap {
      case (path, value) =>
        path.relativeTo(parent).map(child => child -> value)
    }
    Params(childrenParams)
  }
  def +(other: (Path, A)): Params[A] = Params(params + other)
  def -(other: Path): Params[A] = Params(params - other)
  def ++(other: Params[A]): Params[A] =
    Params(params ++ other.params)
  def flatMap[B](f: (Path, A) => Params[B]): Params[B] =
    Params(params.flatMap(f.tupled andThen (_.params)))
  def map[B](f: (Path, A) => (Path, B)): Params[B] =
    Params(params.map(f.tupled))
  def mapValues[B](f: A => B): Params[B] =
    Params(params.map { case (k, v) => (k, f(v)) })
  def filter(f: (Path, A) => Boolean): Params[A] =
    Params(params.filter(f.tupled))
  def filterPaths(f: Path => Boolean): Params[A] =
    Params(params.filter { case (k, _) => f(k) })
  def filterValues(f: A => Boolean): Params[A] =
    Params(params.filter { case (_, v) => f(v) })
  def partition(by: (Path, A) => Boolean): (Params[A], Params[A]) = {
    val (left, right) = params.partition(by.tupled)
    (Params(left), Params(right))
  }
  def partitionPaths(by: Path => Boolean): (Params[A], Params[A]) =
    partition { case (k, _) => by(k) }
  def partitionValues(by: A => Boolean): (Params[A], Params[A]) =
    partition { case (k, v) => by(v) }
  def values: Iterable[A] = params.values
  def join[B](other: Params[B]): Params[(A, B)] = {
    val allPaths = paths ++ other.paths
    val joinedItems = allPaths.map(path => (path, (this(path), other(path))))
    Params(joinedItems.toMap)
  }
  def prefixJoin[B](other: Params[B]): Params[(A, Params[B])] = {
    val joinedItems = paths
      .map { path => path -> ((apply(path), other.children(path))) }
    Params(joinedItems.toMap)
  }
  def size: Int = params.size
  def isEmpty: Boolean = params.isEmpty
  def prependPath(path: Path): Params[A] =
    Params(params.map { case (k, v) => path / k -> v })
  def weights: A = apply(Params.Weights)
}

object Params {
  def apply[A](elems: (Path, A)*): Params[A] =
    new Params[A](Map(elems: _*))
  def empty[A]: Params[A] =
    new Params[A](Map.empty)
  val Weights: Path = "weights"
  val State: Path = "state"
}
