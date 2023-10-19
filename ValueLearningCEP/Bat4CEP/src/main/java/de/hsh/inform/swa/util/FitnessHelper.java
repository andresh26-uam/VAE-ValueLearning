package de.hsh.inform.swa.util;

import java.util.Arrays;
import java.util.Map;

import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.evaluation.EvaluationResult;
import de.hsh.inform.swa.evaluation.RuleEvaluator;
import de.hsh.inform.swa.evaluation.RuleWithFitness;
/**
 * Helper class that passes a set of rules to the CEP engine and updates the rules with the evaluation results received.
 * @author Software Architecture Research Group
 *
 */
public class FitnessHelper {
	public static void measureFitness(RuleEvaluator evaluator, RuleWithFitness... population) {
        Map<Rule, EvaluationResult> s = evaluator.evaluateRule(Arrays.asList(population));
        s.forEach((x, y) -> {
            if (x instanceof RuleWithFitness) {
                ((RuleWithFitness) x).setCondition(y);
            }
        });
    }
}
