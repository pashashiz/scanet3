package scanet.optimizers

trait OptimizersSyntax extends Condition.AllSyntax with SparkExt.AllSyntax

object syntax extends OptimizersSyntax
