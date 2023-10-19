package de.hsh.inform.swa.cep;

import java.util.Objects;
import java.util.Set;

/**
 * Commonality of all parent nodes in the ECT. Mainly logical operation like AND, ->, ...
 * @author Software Architecture Research Group
 *
 */
public abstract class EventCondition implements Condition {
	
	private EventCondition c1, c2;
	
	protected EventCondition(EventCondition c1, EventCondition c2){
		this.c1 = c1;
		this.c2 = c2;
	}
	
    @Override
	public EventCondition[] getSubconditions() {
        return new EventCondition[] { c1, c2 };
    }

    @Override
    public void setSubcondition(Condition newOperand, int position) {
        if (newOperand instanceof EventCondition) {
            if (position == 0) {
                c1 = (EventCondition) newOperand;
            } else if (position == 1) {
                c2 = (EventCondition) newOperand;
            }
        }
    }
    
    @Override
    public int hashCode() {
        return Objects.hash((Object[]) getSubconditions());
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        EventCondition that = (EventCondition) o;
        EventCondition[] thisArr = getSubconditions();
        EventCondition[] thatArr = that.getSubconditions();
        return Objects.equals(thisArr[0], thatArr[0]) && Objects.equals(thisArr[0], thatArr[1]);
    }
	
    public abstract EventCondition copy();

    public abstract String toStringWithAlias(Set<String> aliasesSoFar);
}
