package de.hsh.inform.swa.cep.operators.events;

import java.util.Arrays;
import java.util.Set;

import de.hsh.inform.swa.cep.EventCondition;
/**
 * Entity Class representing an "OR" operation in the ECT. Currently not used in Bat4CEP.
 * @author Software Architecture Research Group
 *
 */
public class OrEventOperator extends EventCondition {

    public OrEventOperator(EventCondition c1, EventCondition c2) {
        super(c1, c2);
    }
    @Override
    public EventCondition copy() {
    	EventCondition[] copy = Arrays.stream(getSubconditions()).map(cond -> cond.copy()).toArray(EventCondition[]::new);
        return new OrEventOperator(copy[0], copy[1]);
    }

    @Override
    public String toString() {
        return String.format("(%s or %s)", Arrays.stream(getSubconditions()).map(c -> c.toString()).toArray());
    }

    @Override
    public String toStringWithAlias(Set<String> aliasesSoFar) {
        return String.format("(%s or %s)", Arrays.stream(getSubconditions()).map(c -> c.toStringWithAlias(aliasesSoFar)).toArray());
    }
}
