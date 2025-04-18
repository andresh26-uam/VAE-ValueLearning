package de.hsh.inform.swa.cep.operators.events;

import java.util.Arrays;
import java.util.Set;

import de.hsh.inform.swa.cep.EventCondition;
/**
 * Entity Class representing a "SEQUENCE" operation in the ECT.
 * @author Software Architecture Research Group
 *
 */
public class SequenceEventOperator extends EventCondition {

    public SequenceEventOperator(EventCondition c1, EventCondition c2) {
        super(c1, c2);
    }

    @Override
    public EventCondition copy() {
    	EventCondition[] copy = Arrays.stream(getSubconditions()).map(cond -> cond.copy()).toArray(EventCondition[]::new);
        return new SequenceEventOperator(copy[0], copy[1]);
    }

    @Override
    public String toString() {
        return String.format("(%s -> %s)", Arrays.stream(getSubconditions()).map(c -> c.toString()).toArray());
    }

    @Override
    public String toStringWithAlias(Set<String> aliasesSoFar) {
        return String.format("(%s -> %s)", Arrays.stream(getSubconditions()).map(c -> c.toStringWithAlias(aliasesSoFar)).toArray());
    }
}
