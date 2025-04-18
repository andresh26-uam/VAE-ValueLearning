package de.hsh.inform.swa.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.Locale;

import de.hsh.inform.swa.bat4cep.util.RunResult;
import de.hsh.inform.swa.evaluation.EvaluationMeasures;
import de.hsh.inform.swa.evaluation.EvaluationResult;
/**
 * Logging class for individual runs of a test case.
 * @author Software Architecture Research Group
 *
 */
public class IndividualLogsCSV {
    private final SimpleLogger csv;
    private int curRun = 1;

    public IndividualLogsCSV(String filename) throws FileNotFoundException {
        csv = SimpleLogger.getSimpleLogger(new File(filename));
        csv.println(";;;;;;Training;;;;Test;;");
        csv.println(";;;;Time;;F1 Score;Recall;Precision;;F1 Score;Recall;Precision");
    }

    public void addRuns(List<RunResult> results) {
        csv.println(curRun++ + ";;;;;;;;;;;");
        int i = 1;
        for (RunResult r : results) {
            csv.print(';');
            csv.print(i++);
            csv.print(';');
            csv.print(r.getRule().toString());
            csv.print(';').print(';');
            csv.print(TimeUtils.formatTime(r.getDuration().getSeconds() * 1000));
            csv.print(';').print(';');
            EvaluationResult res = r.getTrainingsDataResult();

            double precision = EvaluationMeasures.precision(res);
            double recall = EvaluationMeasures.recall(res);
            if (Double.isNaN(precision))
                precision = 0.0f;
            if (Double.isNaN(recall))
                recall = 0.0f;
            csv.print(String.format(Locale.GERMANY, "%.5f;%.5f;%.5f;", EvaluationMeasures.f1Score(res), recall, precision));
            csv.print(';');
            res = r.getTestDataResult();
            precision = EvaluationMeasures.precision(res);
            recall = EvaluationMeasures.recall(res);
            if (Double.isNaN(precision))
                precision = 0.0f;
            if (Double.isNaN(recall))
                recall = 0.0f;
            csv.print(String.format(Locale.GERMANY, "%.5f;%.5f;%.5f;", EvaluationMeasures.f1Score(res), recall, precision));
            csv.println();
        }
        csv.println(";;;;;;;;;;;;");
        csv.flush();
    }

    public void close() {
        csv.flush();
        csv.close();
    }

}
