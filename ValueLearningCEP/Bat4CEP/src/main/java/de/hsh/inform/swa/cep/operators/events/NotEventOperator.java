package de.hsh.inform.swa.cep.operators.events;

import java.util.Objects;
import java.util.Set;

import de.hsh.inform.swa.cep.Condition;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.cep.EventCondition;
/**
 * Entity Class representing an "AND NOT" operation in the ECT.
 * Due to technical details, the NOT operator of Esper did not prove to be suitable for our application. 
 * For this reason, an Esper guard "cep:without" was implemented, which emulates the behavior of an "AND NOT" operator.
 * The following equations are logically equivalent:
 * A -> not B -> C		<=>		(A -> C) and not B		<=>		(A -> C) where cep:without(B)
 * 
 * @author Software Architecture Research Group
 *
 */
public class NotEventOperator extends EventCondition{
	
	private Event negatedEvent;
	
	public NotEventOperator(EventCondition c1, Event negatedEvent) {
		super(c1, null);
		this.negatedEvent = negatedEvent;
	}

	@Override
	public EventCondition copy() {
        return new NotEventOperator(getPattern().copy(), (Event) (negatedEvent.copy()));
	}

	@Override
    public String toString() {
		return toPatternGuard(getPattern().toString());
    }

    @Override
    public String toStringWithAlias(Set<String> aliasesSoFar) {
    	return toPatternGuard(getPattern().toStringWithAlias(aliasesSoFar));
    }
    
    private String toPatternGuard(String pattern) {
    	return String.format("(%s where cep:without(\"%s\"))", pattern, negatedEvent.getType());
    }
    
    private EventCondition getPattern() {
    	return getSubconditions()[0];
    }
    
    @Override
    public EventCondition[] getSubconditions() {
        return new EventCondition[] { super.getSubconditions()[0] };
    }

    @Override
    public void setSubcondition(Condition newOperand, int position) {
        if (position == 0) super.setSubcondition(newOperand, position);
    }
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        NotEventOperator that = (NotEventOperator) o;
        EventCondition thisPattern = getPattern();
        EventCondition thatPattern = that.getPattern();
        return Objects.equals(thisPattern, thatPattern) && Objects.equals(negatedEvent, that.negatedEvent);
    }

}
