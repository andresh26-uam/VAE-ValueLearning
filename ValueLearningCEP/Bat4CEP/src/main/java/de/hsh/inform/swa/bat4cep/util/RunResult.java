package de.hsh.inform.swa.bat4cep.util;

import java.time.Duration;
import java.util.Locale;

import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.evaluation.EvaluationMeasures;
import de.hsh.inform.swa.evaluation.EvaluationResult;
/**
 * Entity class containing the results of a test run
 * @author Software Architecture Research Group
 *
 */
public class RunResult {
    private final Duration duration;
    private final EvaluationResult trainingsDataResult;
    private final EvaluationResult testDataResult;
    private final Rule rule;

    public RunResult(Duration duration, EvaluationResult trainingsDataResult, EvaluationResult testDataResult,
			Rule rule) {
		this.duration = duration;
		this.trainingsDataResult = trainingsDataResult;
		this.testDataResult = testDataResult;
		this.rule = rule;
	}

	private String getResultFor(EvaluationResult res) {
        double precision = EvaluationMeasures.precision(res);
        double recall = EvaluationMeasures.recall(res);
        if (Double.isNaN(precision)) precision = 0.0f;
        if (Double.isNaN(recall)) recall = 0.0f;
        return String.format(Locale.ENGLISH, "F1 Score %.5f (Recall: %.5f Precision: %.5f)", EvaluationMeasures.f1Score(res), recall, precision);
    }

    @Override
    public String toString() {
        return "Duration: " + duration + " | Training data: " + getResultFor(trainingsDataResult) + " | Test data: " + getResultFor(testDataResult) + "| Rule: "
                + rule;
    }

	public Duration getDuration() {
		return duration;
	}

	public EvaluationResult getTrainingsDataResult() {
		return trainingsDataResult;
	}

	public EvaluationResult getTestDataResult() {
		return testDataResult;
	}

	public Rule getRule() {
		return rule;
	}

}
