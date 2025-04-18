package de.hsh.inform.swa.evaluation;

import java.io.Serializable;
import java.util.Locale;

import de.hsh.inform.swa.cep.Rule;
/**
 * A rule that additionally includes their performance. 
 * @author Software Architecture Research Group
 *
 */
public class RuleWithFitness extends Rule implements Comparable<RuleWithFitness>, Serializable {

    private static final long serialVersionUID = 5226453572071274312L;

    public EvaluationResult conditionFitnessResult;

    public RuleWithFitness(Rule rule) {
        this(rule, null);
    }

    public RuleWithFitness(Rule rule, EvaluationResult conditionFitnessResult) {
        super(rule.getEventConditionTreeRoot(), rule.getWindow(), rule.getAction());
        setAttributeConditionTreeRoot(rule.getAttributeConditionTreeRoot());
        this.conditionFitnessResult = conditionFitnessResult;
    }

    @Override
	public RuleWithFitness copy() {
        return new RuleWithFitness(super.copy(), conditionFitnessResult.copy());
    }

    public void setCondition(EvaluationResult eva) {
        this.conditionFitnessResult = eva;
    }

    @Override
    public String toString() {
        if (this.conditionFitnessResult != null) {
            double precision = EvaluationMeasures.precision(conditionFitnessResult);
            double recall = EvaluationMeasures.recall(conditionFitnessResult);
            if (Double.isNaN(precision))
                precision = 0.0f;
            if (Double.isNaN(recall))
                recall = 0.0f;
            return String.format(Locale.ENGLISH, "%.5f (Recall: %.5f Precision: %.5f) --- %s", getTotalFitness(), recall, precision, super.toString());
        } else {
            return super.toString();
        }
    }

    @Override
    public int compareTo(RuleWithFitness o) {
        return Double.compare(getTotalFitness(), o.getTotalFitness());
    }

    public double getTotalFitness() {
        return EvaluationMeasures.f1Score(conditionFitnessResult);
    }
}
