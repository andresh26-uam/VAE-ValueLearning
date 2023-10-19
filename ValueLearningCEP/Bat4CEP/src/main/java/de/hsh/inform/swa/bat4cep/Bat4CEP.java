package de.hsh.inform.swa.bat4cep;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;

import com.espertech.esper.client.Configuration;
import com.espertech.esper.client.EPServiceProvider;
import com.espertech.esper.client.EPServiceProviderManager;

import de.hsh.inform.swa.bat4cep.bat.Bat;
import de.hsh.inform.swa.bat4cep.bat.BatAlgorithm;
import de.hsh.inform.swa.bat4cep.bat.SwarmInitializer;
import de.hsh.inform.swa.bat4cep.bat.initializer.BatHalfAndHalfPopulationInitializer;
import de.hsh.inform.swa.bat4cep.bat.initializer.BatPopulationInitializer;
import de.hsh.inform.swa.bat4cep.bat.update.PointUpdate;
import de.hsh.inform.swa.bat4cep.util.BatConfig;
import de.hsh.inform.swa.bat4cep.util.RunResult;
import de.hsh.inform.swa.cep.Action;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.evaluation.EvaluationMeasures;
import de.hsh.inform.swa.evaluation.EvaluationResult;
import de.hsh.inform.swa.evaluation.RuleEvaluator;
import de.hsh.inform.swa.evaluation.esper.EsperEvaluator;
import de.hsh.inform.swa.evaluation.esper.EventHandlerUtils;
import de.hsh.inform.swa.util.EventHandler;
import de.hsh.inform.swa.util.SimpleLogger;
import de.hsh.inform.swa.util.TimeUtils;
import de.hsh.inform.swa.util.builder.WindowBuilder;

/**
 * Class that initiates the Bat4CEP algorithm. Initialization includes ...
 * ... starting the CEP engine,
 * ... creating the starting population,
 * ... logging,
 * ... triggering the bat algorithm. 
 * @author Software Architecture Research Group 
 *
 */
public class Bat4CEP {
    private static final ChronoUnit WINDOW_TIME_UNIT = ChronoUnit.SECONDS;
    private static final int MIN_WINDOW_LENGTH = 1;
    
    public static List<RunResult> execute(BatConfig config, List<Event> events, List<Event> events_holdout, Event complexEvent, int numRuns,
            int maxECTHeight, int maxACTHeight, SimpleLogger log, int numberOfThreads) {
	
        long totalStartTime = System.currentTimeMillis();
        
        //test data
        EventHandler ehTest = new EventHandler(events_holdout, complexEvent);
        RuleEvaluator validationTest = ruleEvaluatorInit(ehTest, 1); //one thread for test validation
        
        //training data
        EventHandler ehTraining = new EventHandler(events, complexEvent);
        RuleEvaluator validationTraining = ruleEvaluatorInit(ehTraining, numberOfThreads);  
                    
        WindowBuilder wb = new WindowBuilder(MIN_WINDOW_LENGTH, ehTraining.getWithoutComplexEvent().size(), 
        		TimeUtils.getMinimumTimeDistance(ehTraining.getWithoutComplexEvent(), WINDOW_TIME_UNIT),
                TimeUtils.getMaximumTimeDistance(ehTraining.getWithoutComplexEvent(), WINDOW_TIME_UNIT), WINDOW_TIME_UNIT);

         	        
        PointUpdate pu = new PointUpdate(wb, ehTraining.getEventTypes(), ehTraining, maxECTHeight, maxACTHeight);
        
        List<RunResult> runResults = new ArrayList<>();
        
        for (int run = 0; run < numRuns; run++) {
            
            log.println("--- Starting run #" + (run + 1) + " of " + numRuns + " ---");
            long startTime = System.currentTimeMillis();

            BatPopulationInitializer populationInit = new BatHalfAndHalfPopulationInitializer(config.getMaxFrequency(), config.getPulserate(),
                    config.getLoudness(), ehTraining.getEventTypes(), wb, new Action(ehTraining.getComplexEvent()), ehTraining);
            Bat[] swarm = SwarmInitializer.initSwarm(config.getSwarmSize(), true, populationInit, validationTraining, maxECTHeight, maxACTHeight);
            
            try {
                fly(config, swarm, ehTraining, wb, validationTraining, pu, log);
            } catch (InterruptedException | ExecutionException e) {
                System.err.println("Execution failed:" + e.getMessage());
                e.printStackTrace();
                continue;
            }

            Bat bestBat = swarm[0]; //best bat at the first index position

            Duration duration = Duration.ofMillis(System.currentTimeMillis() - startTime);
            log.println("Duration in ISO-8601 format: " + duration.toString());
            
            EvaluationResult testDataResult = validationTest.evaluateRule(bestBat.getSolution());

            double precision = EvaluationMeasures.precision(testDataResult);
            double recall = EvaluationMeasures.recall(testDataResult);
            if (Double.isNaN(precision))
                precision = 0.0f;
            if (Double.isNaN(recall))
                recall = 0.0f;
            log.println(String.format(Locale.ENGLISH, "Cross Validation %.5f (Recall: %.5f Precision: %.5f)", EvaluationMeasures.f1Score(testDataResult),
                    recall, precision));

            runResults.add(new RunResult(duration, bestBat.getSolution().conditionFitnessResult, testDataResult, bestBat.getSolution()));
            log.flush();
            log.println("--- End of run #" + (run + 1) + " of " + numRuns + " ---");
            

        }
        validationTraining.destroy();
        validationTest.destroy();
        
        Duration duration = Duration.ofMillis(System.currentTimeMillis() - totalStartTime);
        log.println("Total Duration in ISO-8601 format: " + duration.toString());

        for (RunResult run : runResults) {
            log.println(run);
        }
        log.flush();
        return runResults;

    }
    /**
     * Initializes the esper engine.
     * @param eventHandler
     * @param number of parallel threads.A good indicator is the number of CPU cores.
     * @return evaluation unit
     */
    private static RuleEvaluator ruleEvaluatorInit(EventHandler eventHandler, int threads) {
        Configuration configuration = EventHandlerUtils.toEsperConfiguration(eventHandler);
        configuration.getEngineDefaults().getThreading().setThreadPoolOutbound(true);
        configuration.getEngineDefaults().getThreading().setThreadPoolOutboundNumThreads(4);	
        configuration.getEngineDefaults().getViewResources().setShareViews(false);
        configuration.getEngineDefaults().getThreading().setListenerDispatchPreserveOrder(false);
        
        EPServiceProvider[] epServiceProviders = new EPServiceProvider[threads];
        for (int i = 0; i < threads; i++) {
            System.out.print("\rInitialization of Esper Engine :" + (int) Math.round(((1.0 * i) / (threads - 1)) * 100) + "%");
            epServiceProviders[i] = EPServiceProviderManager.getProvider(eventHandler.hashCode() + "_Engine" + i, configuration);
        }
        System.out.println();

        return new EsperEvaluator(eventHandler, epServiceProviders);
    }
    

    /**
     * Let the bats fly.
     * @return Sorted list of all bats in the swarm. The better a bat, the lower the index. Therefore, the best bat is located at the first index position.
     */
    private static ArrayList<Bat> fly(BatConfig config, Bat[] swarm, EventHandler eh, WindowBuilder wb, RuleEvaluator re, PointUpdate pu, SimpleLogger log)
            throws InterruptedException, ExecutionException {
        ArrayList<Bat> solutions = new ArrayList<>();
        BatAlgorithm batSwarm = new BatAlgorithm(swarm, eh, wb, re, pu, config.getSwarmSize(), config.getTimesteps(), config.getLoudness(), config.getAlpha(),
                config.getMaxFrequency(), config.getMinFrequency(), config.getPulserate(), config.getGamma(), log);
        solutions.addAll(batSwarm.execute());
        return solutions;
    }
}