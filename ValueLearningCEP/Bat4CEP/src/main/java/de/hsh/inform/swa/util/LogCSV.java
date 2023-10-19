package de.hsh.inform.swa.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.Locale;
import java.util.Map.Entry;

import de.hsh.inform.swa.bat4cep.BatTestCase;
import de.hsh.inform.swa.bat4cep.util.RunResult;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.evaluation.EvaluationMeasures;
import de.hsh.inform.swa.util.data.AttributeConfig;
import de.hsh.inform.swa.util.data.DataCreatorConfig;
/**
 * Logging class for whole test cases.
 * @author Software Architecture Research Group
 *
 */
public class LogCSV {
    private final SimpleLogger csv;
    private int curRun = 1;

    public LogCSV(String filename) throws FileNotFoundException {
        csv = SimpleLogger.getSimpleLogger(new File(filename));
        csv.println(";Rule;Result;;;;;Training;;;;Test;;;;HITS;;;;Data;;;;;;;;Swarms;;;;;;;\r\n" + ";;# of runs:;;\"total\r\n" + "time\";\"avg\r\n"
                + "time \";;\"avg \r\n" + "F1 Score\";\"avg \r\n" + "recall\";\"avg  \r\n" + "precision\";;\"avg \r\n" + "F1 Score\";\"avg \r\n"
                + "recall\";\"avg  \r\n" + "precision\";;;Training;Test;;Events;\"Event\r\n" + "Types\";\"Event\r\n" + "attributes\";\"Min sec \r\n"
                + "between events\";\"Max sec \r\n" + "between events\";\"Complex \r\n"
                + "Event\";Event attributes;;;Timestamps;Size;Alpha;Gamma;Loudness;\"Max \r\n" + "frequency\";Pulsrate");
    }


    public void addRuns(BatTestCase t, List<RunResult> results, List<Event> events, List<Event> eventsCV) {
        DataCreatorConfig dataCreatorConfig = t.getTrainingData();

        csv.print(curRun++).print(';');
        csv.print(dataCreatorConfig.getRule().toString());
        csv.print(';');
        csv.print(t.getNumRuns());
        csv.print(';').print(';');

        long totalSec = results.stream().mapToLong(r -> r.getDuration().getSeconds()).sum();
        csv.print(TimeUtils.formatTime(totalSec * 1000));
        csv.print(';');
        csv.print(TimeUtils.formatTime((totalSec / t.getNumRuns()) * 1000));
        csv.print(';').print(';');
        csv.print(String.format(Locale.GERMANY, "%.5f",
                results.stream().mapToDouble(r -> EvaluationMeasures.f1Score(r.getTrainingsDataResult())).average().orElse(0.0)))
           .print(';');
        csv.print(String.format(Locale.GERMANY, "%.5f",
                results.stream().mapToDouble(r -> EvaluationMeasures.recall(r.getTrainingsDataResult())).average().orElse(0.0)))
           .print(';');
        csv.print(String.format(Locale.GERMANY, "%.5f",
                results.stream().mapToDouble(r -> EvaluationMeasures.precision(r.getTrainingsDataResult())).average().orElse(0.0)))
           .print(';');
        csv.print(';');
        csv.print(String.format(Locale.GERMANY, "%.5f",
                results.stream().mapToDouble(r -> EvaluationMeasures.f1Score(r.getTestDataResult())).average().orElse(0.0)))
           .print(';');
        csv.print(String.format(Locale.GERMANY, "%.5f",
                results.stream().mapToDouble(r -> EvaluationMeasures.recall(r.getTestDataResult())).average().orElse(0.0)))
           .print(';');
        csv.print(String.format(Locale.GERMANY, "%.5f",
                results.stream().mapToDouble(r -> EvaluationMeasures.precision(r.getTestDataResult())).average().orElse(0.0)))
           .print(';');
        csv.print(';');
        csv.print(';');
        csv.print(new EventHandler(events, t.getTrainingData().getComplexEvent()).getComplexEventCount()).print(';');
        csv.print(new EventHandler(eventsCV, t.getTrainingData().getComplexEvent()).getComplexEventCount()).print(';');
        csv.print(';');
        csv.print(dataCreatorConfig.getNumEvents()).print(';');
        csv.print(dataCreatorConfig.getNumEventTypes()).print(';');
        csv.print(dataCreatorConfig.getNumEventAttributes()).print(';');
        csv.print(dataCreatorConfig.getMinIntermediateSeconds()).print(';');
        csv.print(dataCreatorConfig.getMaxIntermediateSeconds()).print(';');
        csv.print(dataCreatorConfig.getComplexEvent().getType()).print(';');

        for (Entry<String, AttributeConfig<?>> entry : dataCreatorConfig.getFixedSensorRange().entrySet()) {
            csv.print(entry.getKey() + ": {" + entry.getValue().getMinValue() + " - " + entry.getValue().getMaxValue() + "}, ");
        }
        for (int i = dataCreatorConfig.getFixedSensorRange().size(); i < dataCreatorConfig.getNumEventAttributes(); i++) {
            csv.print("OTHER_" + i + " : {" + dataCreatorConfig.getDefaultRange().getMinValue() + " - " + dataCreatorConfig.getDefaultRange().getMaxValue()
                    + "}, ");
        }
        csv.print(';');
        csv.print(';');
        csv.print(';');

        csv.print(String.format(Locale.GERMANY, "%d", t.getBatConfig().getTimesteps())).print(';');
        csv.print(String.format(Locale.GERMANY, "%d", t.getBatConfig().getSwarmSize())).print(';');
        csv.print(String.format(Locale.GERMANY, "%.2f", t.getBatConfig().getAlpha())).print(';');
        csv.print(String.format(Locale.GERMANY, "%.2f", t.getBatConfig().getGamma())).print(';');
        csv.print(String.format(Locale.GERMANY, "%.2f", t.getBatConfig().getLoudness())).print(';');
        csv.print(String.format(Locale.GERMANY, "%.2f", t.getBatConfig().getMaxFrequency())).print(';');
        csv.print(String.format(Locale.GERMANY, "%.2f", t.getBatConfig().getPulserate())).print(';');

        csv.println();
        csv.flush();
    }

    public void close() {
        csv.close();
        csv.flush();

    }
}
