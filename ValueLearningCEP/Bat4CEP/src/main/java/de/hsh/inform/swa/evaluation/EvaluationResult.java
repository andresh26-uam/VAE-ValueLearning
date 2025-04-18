package de.hsh.inform.swa.evaluation;

/**
 * Entity Class representing an evaluation result. I.e. contains information how good a found rule is.
 * @author Software Architecture Research Group
 *
 */
public class EvaluationResult {
	private final long truePositives, falsePositives, trueNegatives, falseNegatives, originalPositives;

    public EvaluationResult(int truePositives, int falsePositives, long trueNegatives, long falseNegatives, long complexEventCount) {
        this.truePositives = truePositives;
        this.falsePositives = falsePositives;
        this.trueNegatives = trueNegatives;
        this.falseNegatives = falseNegatives;
        this.originalPositives = complexEventCount; //for a perfect result:  originalPositives == truePositives AND falseNegatives == 60
    }

    public EvaluationResult copy() {
        return new EvaluationResult((int) truePositives, (int) falsePositives, trueNegatives, falseNegatives, originalPositives);
    }
    
    public long getTruePositives() {
		return truePositives;
	}

	public long getFalsePositives() {
		return falsePositives;
	}

	public long getTrueNegatives() {
		return trueNegatives;
	}

	public long getFalseNegatives() {
		return falseNegatives;
	}

	public long getOriginalPositives() {
		return originalPositives;
	}
}
