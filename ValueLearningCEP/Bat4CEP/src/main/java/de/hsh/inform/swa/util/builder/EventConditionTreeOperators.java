package de.hsh.inform.swa.util.builder;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.BiFunction;

import de.hsh.inform.swa.cep.Condition;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.cep.EventCondition;
import de.hsh.inform.swa.cep.operators.events.AndEventOperator;
import de.hsh.inform.swa.cep.operators.events.NotEventOperator;
import de.hsh.inform.swa.cep.operators.events.SequenceEventOperator;
/**
 * Class that defines all legal operands in the ECT (currently AND and SEQUENCE).
 * @author Software Architecture Research Group
 *
 */
public class EventConditionTreeOperators {
	private static final List<BiFunction<EventCondition, EventCondition, EventCondition>> logicalOperatorList = new ArrayList<>();
	
	static{
        logicalOperatorList.add((left, right) -> new AndEventOperator(left, right));
        logicalOperatorList.add((left, right) -> new SequenceEventOperator(left, right));
//        logicalOperatorList.add((left, right) -> new OrEventOperator(left, right));		//currently not used in Bat4CEP
    }
	public static EventCondition getRandomLogicalOperator(EventCondition left, EventCondition right) {
    	int randomIdx = ThreadLocalRandom.current().nextInt(logicalOperatorList.size());
        return logicalOperatorList.get(randomIdx).apply(left, right);
    }
	public static EventCondition updateNotOperator(Event notEvent, Condition originalNode) {
		if(!(originalNode instanceof NotEventOperator)){
			return new NotEventOperator((EventCondition) originalNode, notEvent);
    	}else {
    		return new NotEventOperator((EventCondition) originalNode.getSubconditions()[0], notEvent);
    	}
    }
}