package de.hsh.inform.swa.evaluation;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import de.hsh.inform.swa.cep.Rule;
/**
 * Commonality of all evaluation classes.
 * @author Software Architecture Research Group
 *
 */
public interface RuleEvaluator {
	/**
     * method passes a rule to the engine, where it is subsequently applied to the event stream.
     * @param rules
     * @returnRules Rule with information about their performance
     */
    default public EvaluationResult evaluateRule(RuleWithFitness rule) {
    	return evaluateRule(Arrays.asList(rule)).get(rule);
    }

    /**
     * passes rules to the engine, which are then applied to the event stream.
     * @param rules
     * @return Rules with information about their performance
     */
    public Map<Rule, EvaluationResult> evaluateRule(List<Rule> rules);

    default public void destroy() {}
}
