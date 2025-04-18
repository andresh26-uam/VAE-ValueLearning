package de.hsh.inform.swa.evaluation.esper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import com.espertech.esper.client.EPRuntime;
import com.espertech.esper.client.EPServiceProvider;
import com.espertech.esper.client.EPStatement;
import com.espertech.esper.client.time.CurrentTimeEvent;
import com.espertech.esper.client.time.TimerControlEvent;
import com.google.common.collect.Lists;

import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.evaluation.EvaluationResult;
import de.hsh.inform.swa.evaluation.EvaluationSubscriber;
import de.hsh.inform.swa.evaluation.RuleEvaluator;
import de.hsh.inform.swa.util.EventHandler;
import de.hsh.inform.swa.util.builder.AttributeConditionTreeBuilder;
/**
 * This class determines the performance of new rules and uses the CEP engine Esper. 
 * Depending on how many threads are defined,
 * this class splits the rules into subsets and evaluates them in parallel.
 * 
 * @author Software Architecture Research Group
 *
 */
public class EsperEvaluator implements RuleEvaluator {
    private final List<EPServiceProvider> esperServiceProvider = new ArrayList<>();
    private final AtomicInteger curStart = new AtomicInteger();
    private final EventHandler eh;
        
    public EsperEvaluator(EventHandler eh, EPServiceProvider[] esperServiceProvider) {
        this.eh = eh;
        this.esperServiceProvider.addAll(Arrays.asList(esperServiceProvider));
    }

    @Override
    public Map<Rule, EvaluationResult> evaluateRule(List<Rule> inRules) {
    	initPatternGuards();
        int size = (inRules.size() / esperServiceProvider.size());

        List<List<Rule>> subSets = Lists.partition(inRules, size); //external library: google.common

        AtomicInteger curProviderID = new AtomicInteger(curStart.getAndIncrement());
        if (curStart.get() >= esperServiceProvider.size()) {
            curStart.set(0);
        }

        return subSets.stream().parallel().flatMap(curRules -> {
            int curID = curProviderID.getAndIncrement();
            if (curID >= esperServiceProvider.size()) {
                curID = 0;
                curProviderID.set(0);
            }
            final EPServiceProvider provider = esperServiceProvider.get(curID);
            /*
             * We need to make sure the engine is only used once at a time.
             * 
             * Workflow for the following part:
             * 1. Create rules. Each rule has a registered subscriber which gets notified by the CEP engine 
             * 2. Feed the CEP engine with events 
             * 3. Collect the results.
             */
            synchronized (provider) {
            	provider.initialize();
                // Create a bunch of statements, one for each rule and register a subscriber
                Map<Rule, EPStatement> statements = curRules.stream().parallel().collect(Collectors.toMap(rule -> rule, rule -> {
                	AttributeConditionTreeBuilder.repairAggregationWindowsInAct(rule, eh); // special treatment because of Esper-specific aggregation functions
                	EPStatement statement = EsperUtils.createStatement(provider.getEPAdministrator(), rule);
                    EvaluationSubscriber subscriber = new EvaluationSubscriber(eh);
                    statement.setSubscriber(subscriber);
                    statement.start();
                    return statement;
                }));  
                
                // Feed the engine with data
                sendEvents(provider.getEPRuntime(), eh.getEventData());
                // Collect the results.
                Map<Rule, EvaluationResult> results = statements.entrySet().stream().parallel().collect(Collectors.toMap(res -> res.getKey(), res -> {

                    EPStatement statement = res.getValue();
                    EvaluationSubscriber subscriber = (EvaluationSubscriber) statement.getSubscriber();

                    int truePositives = subscriber.getTruePositives();
                    int falsePositives = subscriber.getFalsePositives();

                    long trueNegatives = (eh.getEventDataSize() - eh.getComplexEventCount()) - falsePositives;
                    long falseNegatives = eh.getComplexEventCount() - truePositives;

                    if(!statement.isDestroyed()) {
                    	statement.destroy();
                    }
                    
                    return new EvaluationResult(truePositives, falsePositives, trueNegatives, falseNegatives, eh.getComplexEventCount());
                }));
                return results.entrySet().stream();
            }
        }).collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue()));
    }

    private void sendEvents(EPRuntime runtime, List<Event> events) {
    	runtime.sendEvent(new TimerControlEvent(TimerControlEvent.ClockType.CLOCK_EXTERNAL));
        for (Event e : events) {
        	// see esper doc: "Catching up a Statement from Historical Data"
        	// https://www.espertech.com/esper/esper-documentation/
            runtime.sendEvent(new CurrentTimeEvent(e.getTimestamp()));
            runtime.sendEvent(e.getAttributes(), e.getType());
        }
    }
    
    private void initPatternGuards() {
    	WithoutGuard.initEvents(eh.getEventData());	//one pattern guard so far
    }

    @Override
    public void destroy() {
        for (EPServiceProvider ep : esperServiceProvider) {
            ep.destroy();
        }
        esperServiceProvider.clear();
    }
}
