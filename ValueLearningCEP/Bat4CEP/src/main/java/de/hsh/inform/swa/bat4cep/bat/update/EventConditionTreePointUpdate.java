package de.hsh.inform.swa.bat4cep.bat.update;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.BiFunction;

import de.hsh.inform.swa.cep.Condition;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.cep.EventCondition;
import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.cep.operators.events.NotEventOperator;
import de.hsh.inform.swa.util.EventHandler;
import de.hsh.inform.swa.util.builder.AttributeConditionTreeBuilder;
import de.hsh.inform.swa.util.builder.ConditionTreeTraverser;
import de.hsh.inform.swa.util.builder.EventConditionTreeOperators;
/**
 * Class responsible for updates to the ECT of a given rule.
 * These updates include changes to ...
 * ... logical operators (not, and, ->)
 * ... events.
 * 
 * @author Software Architecture Research Group
 *
 */
public class EventConditionTreePointUpdate {

    private final List<Event> eventTypes = new ArrayList<>();
    private final List<BiFunction<EventCondition, EventCondition, EventCondition>> eventConditionBuilderList = new ArrayList<>();
    
    public EventConditionTreePointUpdate(List<Event> eventTypes) {
        this.eventTypes.addAll(eventTypes);
        eventConditionBuilderList.add((left, right) -> getRandomEvent());
        eventConditionBuilderList.add((left, right) -> getRandomLogicalOperator(left, right));
    }
    
    private EventCondition getRandomEventCondition(EventCondition left, EventCondition right, Condition originalNode) {
    	int randomIdx = ThreadLocalRandom.current().nextInt(eventConditionBuilderList.size() + 1);
    	if(randomIdx == eventConditionBuilderList.size()) return EventConditionTreeOperators.updateNotOperator(getRandomEvent(), originalNode);
        return eventConditionBuilderList.get(randomIdx).apply(left, right);
    }
    
    public Event getRandomEvent() {
        int idx = ThreadLocalRandom.current().nextInt(eventTypes.size());
        return eventTypes.get(idx);
    }
    
    public void update(Rule rule, int updatePoint, EventHandler eh) {
        Condition originalNode = ConditionTreeTraverser.getConditionWithPreOrderIndex(rule.getEventConditionTreeRoot(), updatePoint);
        EventCondition[] operands = (EventCondition[]) originalNode.getSubconditions();
        List<EventCondition> newOperands = getNumberOfOperands(operands, 2);
        
        EventCondition newNode = getRandomEventCondition(newOperands.get(0), newOperands.get(1), originalNode);
        rule.setEventConditionTreeRoot(ConditionTreeTraverser.replaceNode(rule.getEventConditionTreeRoot(), newNode, updatePoint));
        repairECT(rule);
        AttributeConditionTreeBuilder.repairAct(rule, eh);
    }
    
    /**
     * this method iterates the ECT and removes all one-element NOT conditions like "A and not A" or "A and not B".
     * More precisely, the right part of an "AND NOT" relation is getting removed. E.g. "((A and not B) and not C)" becomes "A"
     * @param rule
     */
    public void repairECT(Rule rule) {
    	int pos=0;
    	while(ConditionTreeTraverser.getConditionWithPreOrderIndex(rule.getEventConditionTreeRoot(), pos) != null) {
    		Condition originalNode = ConditionTreeTraverser.getConditionWithPreOrderIndex(rule.getEventConditionTreeRoot(), pos);
    		if(originalNode instanceof NotEventOperator) {
    			if(originalNode.getSubconditions()[0] instanceof Event) {
    				if(pos==0) {
    					rule.setEventConditionTreeRoot((EventCondition) originalNode.getSubconditions()[0]);
    				}else {
    					ConditionTreeTraverser.replaceNode(rule.getEventConditionTreeRoot(), originalNode.getSubconditions()[0], pos);
        				pos=-1; //restart from scratch to exclude recursive problems
    				}
    				
    			}
    		}
    		pos++;
    	}
    }

	public void reduceComplexity(Rule rule, int updatePoint, EventHandler eh) {
        Condition originalNode = ConditionTreeTraverser.getConditionWithPreOrderIndex(rule.getEventConditionTreeRoot(), updatePoint);
        if (!(originalNode instanceof Event))
            rule.setEventConditionTreeRoot(ConditionTreeTraverser.replaceNode(rule.getEventConditionTreeRoot(), getRandomEvent(), updatePoint));
        AttributeConditionTreeBuilder.repairAct(rule, eh);
    }

    private List<EventCondition> getNumberOfOperands(EventCondition[] oldOperands, int numberOfOperandsNeeded) {
        List<EventCondition> newOperands = new ArrayList<>();
        if (oldOperands != null) {
            newOperands.addAll(Arrays.asList(oldOperands));
        }
        while (newOperands.size() < numberOfOperandsNeeded) {
            newOperands.add(getRandomEvent());
        }
        return newOperands;
    }
    public int getNumberOfEventTypes() {
    	return eventTypes.size();
    }

    //delegation
	public EventCondition getRandomLogicalOperator(EventCondition left, EventCondition right) {
		return EventConditionTreeOperators.getRandomLogicalOperator(left, right);
	}
}
