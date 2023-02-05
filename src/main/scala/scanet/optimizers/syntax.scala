package scanet.optimizers

trait OptimizersSyntax extends Condition.AllSyntax with SparkExt.AllSyntax with Iterators.AllSyntax

object syntax extends OptimizersSyntax
