package de.hsh.inform.swa.util.data;

import java.security.SecureRandom;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import com.espertech.esper.client.Configuration;
import com.espertech.esper.client.EPServiceProvider;
import com.espertech.esper.client.EPServiceProviderManager;
import com.espertech.esper.client.EPStatement;
import com.espertech.esper.client.time.CurrentTimeEvent;
import com.espertech.esper.client.time.TimerControlEvent;
import com.espertech.esper.collection.Pair;

import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.cep.windows.LengthWindow;
import de.hsh.inform.swa.cep.windows.TimeWindow;
import de.hsh.inform.swa.cep.windows.Window;
import de.hsh.inform.swa.evaluation.EvaluationMeasures;
import de.hsh.inform.swa.evaluation.EvaluationResult;
import de.hsh.inform.swa.evaluation.EvaluationSubscriber;
import de.hsh.inform.swa.evaluation.esper.WithoutGuard;
import de.hsh.inform.swa.evaluation.esper.EsperSubscriber;
import de.hsh.inform.swa.evaluation.esper.EsperUtils;
import de.hsh.inform.swa.evaluation.esper.EventHandlerUtils;
import de.hsh.inform.swa.util.EventHandler;
import de.hsh.inform.swa.util.builder.AttributeConditionTreeBuilder;
/**
 * This class initiates the data generation.
 * @author Software Architecture Research Group
 *
 */
public class DataCreator {

    private static final Event COMPLEX_EVENT = new Event("HIT");

    private static List<Event> createEvents(DataCreatorConfig config) {
    	
        String[] types = new String[config.getNumEventTypes()];
        Event[] eventTypes = new Event[types.length];
        Map<String, Integer> numberOfAttributesOfType = null;

        for (int i = 0; i < types.length; i++) {
            types[i] = Character.toString((char) ('A' + i % 26));
            for (int j = 0; j < (i / 26); j++) {
                types[i] += types[i].substring(0, 1);
            }
        }

        eventTypes = new Event[types.length];
        for (int i = 0; i < eventTypes.length; i++) {
            eventTypes[i] = new Event(types[i]);
        }

        numberOfAttributesOfType = getMapOfTypeToNumberOfAttributes(types, config.getNumEventAttributes(), config.getNumEventAttributes());

        int eventNumber = 0;

        List<Event> eventList = new ArrayList<>();
        Date startDate = Date.from(Instant.now());
        // keep good portions of random generated data which have at least one hit with
        // the length of the rule window.
        do {
        	/*
        	 * data generation idea:
        	 * 1. Create a random event stream ten times larger than the specified stream size. When 15,000 events are set, 150,000 events are created first.
        	 * 2. Determine all the places where the rule fires (HIT).
        	 * 3. Save all events involved in a hit until the maximum event size is reached.
        	 * 
        	 * This has the advantage of having as many HITS as possible in the event stream to prevent underfitting.
        	 */
        	int accelerationFactor = config.getHitPercentage()==1.0 ? 10: 2;
            List<Event> withoutComplex = createData(types, config.getNumEvents() * accelerationFactor, config.getNumEventAttributes(), startDate,
                    config.getMinIntermediateSeconds(), config.getMaxIntermediateSeconds(), numberOfAttributesOfType, config.getDefaultRange(),
                    config.getFixedSensorRange());
            
            System.out.println(config.getRule());
            List<Event> result = addComplexEventsToStream(withoutComplex, config.getRule());

            Window window = config.getRule().getWindow();
            for (int i = result.size() - 1; i >= 0; i--) {
                Event curEvent = result.get(i);
                if (curEvent.getType().equals(config.getComplexEvent().getType())) {
                    int count = 0;
                    long timeWindowstart = result.get(i - 1).getTimestamp();
                    boolean breakall = false;

                    for (int j = i - 1; j >= 0; j--) {
                        Event prevEvent = result.get(j);
                        
                        if (!prevEvent.getType().equals(config.getComplexEvent().getType())) {
                            eventList.add(prevEvent);
                            eventNumber++;
                            if (eventNumber > config.getNumEvents()) {
                                breakall = true;
                                break;
                            }
                        }
                        long windowValue = window.getValue();
                        if (window instanceof TimeWindow) {
                            long timediff = timeWindowstart - prevEvent.getTimestamp();
                            long windowValueInMS = windowValue * 1000;
                            if (prevEvent.getType().equals(config.getComplexEvent().getType()) && timediff < windowValueInMS) {
                                timeWindowstart = prevEvent.getTimestamp();
                                continue;
                            }

                            if (timediff > windowValueInMS) {
                                i = j - 1;
                                break;
                            }

                        } else if (window instanceof LengthWindow) {
                            if (result.get(j).getType().equals(config.getComplexEvent().getType()) && count++ < windowValue) {
                                count = 0;
                                continue;
                            }
                            if (count > windowValue) {
                                i = j - 1;
                                break;
                            }
                        }
                    }
                    if (breakall) {
                        break;
                    }
                }
            }
            System.out.print(String.format("\rGenerating dataset: %.2f%%", (eventNumber * 1.0 / config.getNumEvents()) * 100));
            startDate = Date.from(result.get(result.size() - 1).getTime().toInstant().plusSeconds(1));
        } while (eventNumber < config.getNumEvents()*config.getHitPercentage());
        
        
        if(eventNumber < config.getNumEvents()) {
        	List<Event> withoutComplex = createData(types, config.getNumEvents()*10, config.getNumEventAttributes(), startDate,
                    config.getMinIntermediateSeconds(), config.getMaxIntermediateSeconds(), numberOfAttributesOfType, config.getDefaultRange(),
                    config.getFixedSensorRange());
        	List<Event> result = addComplexEventsToStream(withoutComplex, config.getRule());
        	
        	int idx= result.size();
            while(eventNumber < config.getNumEvents()) {
            	idx--;
            	Event ev = result.get(idx);
            	if(ev.getType().equals(config.getComplexEvent().getType())){
            		idx--;
            	}else {
            		eventList.add(ev);
            		eventNumber++;
            	}
            }
        	
        }
        
        System.out.println();

        Collections.sort(eventList);
        
        // fix timestamps
        long delta = 0;
        for (int i = 0; i < eventList.size(); i++) {
            Event cur = eventList.get(i);
            if (i + 1 < eventList.size()) {
                Event next = eventList.get(i + 1);
                next.setTime(Date.from(next.getTime().toInstant().plusSeconds(-delta)));
                if (cur.getTime().toInstant().until(next.getTime().toInstant(), ChronoUnit.SECONDS) > config.getMaxIntermediateSeconds()) {
                    long newdelta = (next.getTimestamp() - cur.getTimestamp()) / 1000; // millisec to sec
                    int addedTime;
                    if(config.getMinIntermediateSeconds() < config.getMaxIntermediateSeconds()) {
                        addedTime = ThreadLocalRandom.current().nextInt(config.getMinIntermediateSeconds(), config.getMaxIntermediateSeconds()); // also add random delay
                    }else {
                        addedTime = config.getMinIntermediateSeconds();
                    }
                    newdelta -= addedTime;
                    next.setTime(Date.from(next.getTime().toInstant().plusSeconds(-newdelta)));
                    delta += newdelta;
                }
            }
        }
        // get final HITs
        List<Event> result = addComplexEventsToStream(eventList.subList(0, config.getNumEvents()), config.getRule());
        
        return result;
    } 	

    private static Map<String, Integer> getMapOfTypeToNumberOfAttributes(String[] types, int minAttributes, int maxAttributes) {
        Map<String, Integer> numberOfAttributesOfType = new HashMap<>();
        Random rnd = new Random();
        int rndInterval = maxAttributes - minAttributes + 1;
        for (int i = 0; i < types.length; i++) {
            numberOfAttributesOfType.put(types[i], minAttributes + rnd.nextInt(rndInterval));
        }
        return numberOfAttributesOfType;
    }
    
    private static List<Event> createData(String[] eventTypes, int numberOfEvents, int numberOfEventAttributes, Date startDate,
            int minIntermediateSeconds, int maxIntermediateSeconds, Map<String, Integer> numberOfAttributesOfType, AttributeConfig<?> defaultRange,
            Map<String, AttributeConfig<?>> fixedSensorRange) {
    		ArrayList<Event> generatedEvents = new ArrayList<>(numberOfEvents);
    		SecureRandom random = new SecureRandom();
    		for (int i = 0; i < numberOfEvents; i++) {
    			int addedTime = minIntermediateSeconds;
    			if(minIntermediateSeconds < maxIntermediateSeconds) {
    				addedTime = addedTime + random.nextInt(maxIntermediateSeconds - minIntermediateSeconds);
    			}
	            
	            Date time = Date.from(startDate.toInstant().plusSeconds(addedTime));
	            startDate = time;
	            int typePos = random.nextInt(eventTypes.length);
	            String type = eventTypes[typePos];
	            Map<String, Object> attributes = createAttributeValues(numberOfAttributesOfType.get(type), numberOfEventAttributes, defaultRange, fixedSensorRange);
	            Event event = new Event(type, time, attributes);
	            generatedEvents.add(event);
    		}

        return generatedEvents;
    }

    public static double eval(List<Event> events, Rule rule) {
    	EventHandler eh = new EventHandler(events, COMPLEX_EVENT);
        
        Configuration configuration = EventHandlerUtils.toEsperConfiguration(eh);

        EPServiceProvider epServiceProvider = EPServiceProviderManager.getDefaultProvider(configuration);        
        
        EPStatement statement = EsperUtils.createStatement(epServiceProvider.getEPAdministrator(), rule);
        EvaluationSubscriber subscriber = new EvaluationSubscriber(eh);
        statement.setSubscriber(subscriber);
        statement.start();

        epServiceProvider.getEPRuntime().sendEvent(new TimerControlEvent(TimerControlEvent.ClockType.CLOCK_EXTERNAL));

        for (Event event : events) {
            // esper doc: Catching up a Statement from Historical Data
            epServiceProvider.getEPRuntime().sendEvent(new CurrentTimeEvent(event.getTimestamp()));
            epServiceProvider.getEPRuntime().sendEvent(event.getAttributes(), event.getType());
        }

        int truePositives = subscriber.getTruePositives();
        int falsePositives = subscriber.getFalsePositives();

        long trueNegatives = (eh.getEventDataSize() - eh.getComplexEventCount()) - falsePositives;
        long falseNegatives = eh.getComplexEventCount() - truePositives;

        statement.destroy();
        epServiceProvider.destroy();
        
        return EvaluationMeasures.f1Score(new EvaluationResult(truePositives, falsePositives, trueNegatives, falseNegatives, falseNegatives));
    }
    
    static List<Event> addComplexEventsToStream(List<Event> generatedEvents, Rule rule) {
    	
    	AttributeConditionTreeBuilder.repairAggregationWindowsInAct(rule, null);
        List<Event> result = new ArrayList<>();
        result.addAll(generatedEvents);
        Event[] events = generatedEvents.toArray(new Event[0]);
        EventHandler eh = new EventHandler(generatedEvents, COMPLEX_EVENT);
        
        Configuration configuration = EventHandlerUtils.toEsperConfiguration(eh);
        WithoutGuard.initEvents(eh.getEventData());

        EPServiceProvider epServiceProvider = EPServiceProviderManager.getDefaultProvider(configuration);        
        
        EPStatement statement = EsperUtils.createStatement(epServiceProvider.getEPAdministrator(), rule);
        EsperSubscriber subscriber = new EsperSubscriber();
        System.out.println(subscriber);
        statement.setSubscriber(subscriber);
        statement.start();

        epServiceProvider.getEPRuntime().sendEvent(new TimerControlEvent(TimerControlEvent.ClockType.CLOCK_EXTERNAL));

        for (int i = 0; i < events.length; i++) {
            events[i].setNumber(i);
        }

        for (Event event : events) {
            // esper doc: Catching up a Statement from Historical Data
            epServiceProvider.getEPRuntime().sendEvent(new CurrentTimeEvent(event.getTimestamp()));
            epServiceProvider.getEPRuntime().sendEvent(event.getAttributes(), event.getType());
        }

        statement.destroy();
        epServiceProvider.destroy();

        Set<Integer> firedAt = subscriber.getFiredPosition();

        int offset = 1;
        for (int curVal : firedAt) {
            Date ldt = Date.from(result.get(curVal + offset - 1).getTime().toInstant().plusSeconds(0));

            result.add(curVal + offset, new Event(rule.getAction().getComplexEvent().getType(), ldt));
            offset++;
        }

        for (int i = 0; i < result.size(); i++) {
            result.get(i).setNumber(i);
        }
        
        return result;
    }

    private static Map<String, Object> createAttributeValues(int numberOfAttributes, int numberOfSensors, AttributeConfig<?> defaultRange,
            Map<String, AttributeConfig<?>> fixedSensorRange) {
        Map<String, Object> attributes = new HashMap<>();

        for (Entry<String, AttributeConfig<?>> e : fixedSensorRange.entrySet()) {
            attributes.put(e.getKey(), e.getValue().getRandomValue());
        }

        for (int i = attributes.size(); attributes.size() < numberOfAttributes; i++) {
            attributes.put("OTHER_" + i++, defaultRange.getRandomValue());
        }

        return attributes;
    }
    public static Pair<List<Event>,List<Event>> getEvents(DataCreatorConfig config) {
    	List<Event> events, holdoutEvents;
    	if(config.getData() == DatasetEnum.SYNTHETIC) {
    		events = createEvents(config);
    		holdoutEvents = createEvents(config);
    	} else { //traffic data set
    		Pair<List<Event>, List<Event>> allEvents = DataCreatorTraffic.getTrafficEventsWithHits(config);
    		events = allEvents.getFirst();
    		holdoutEvents = allEvents.getSecond();
    	}
    	return new Pair<List<Event>, List<Event>>(events, holdoutEvents);
    }
}
