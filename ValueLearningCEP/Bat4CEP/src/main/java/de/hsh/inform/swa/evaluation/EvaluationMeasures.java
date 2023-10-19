package de.hsh.inform.swa.evaluation;
/**
 * Helper class that computes key machine learning metrics from a given evaluation result. 
 * @author Software Architecture Research Group
 *
 */
public class EvaluationMeasures {
    public static double truePositiveRate(EvaluationResult e) {
        if (e.getTruePositives() + e.getFalseNegatives() == 0.0) {
            return 0.0;
        }
        return (double) e.getTruePositives() / (e.getTruePositives() + e.getFalseNegatives());
    }

    public static double precision(EvaluationResult e) {
        if (e.getTruePositives() + e.getFalsePositives() == 0.0) {
            return 0.0;
        }
        return (double) e.getTruePositives() / (e.getTruePositives() + e.getFalsePositives());
    }

    public static double recall(EvaluationResult e) {
        if (e.getTruePositives() + e.getFalseNegatives() == 0.0) {
            return 0.0;
        }
        return (double) e.getTruePositives() / (e.getTruePositives() + e.getFalseNegatives());
    }
    public static double f1Score(EvaluationResult e) {
        double precision = precision(e);
        double recall = truePositiveRate(e);
        if (precision + recall == 0.0) {
            return 0.0;
        }
        return 2.0 * (precision * recall) / (precision + recall);
    }
}
