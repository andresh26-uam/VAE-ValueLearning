package de.hsh.inform.swa.cep.operators.attributes.logic;

import java.util.Arrays;

import de.hsh.inform.swa.cep.AttributeCondition;

/**
 * Entity Class representing an "AND" operation in the ACT.
 * @author Software Architecture Research Group
 *
 */
public class AndAttributeOperator extends AttributeCondition {

    public AndAttributeOperator(AttributeCondition cond1, AttributeCondition cond2) {
        super(cond1, cond2);
    }

    @Override
    public AttributeCondition copy() {
    	AttributeCondition[] copy = Arrays.stream(getSubconditions()).map(cond -> cond.copy()).toArray(AttributeCondition[]::new);
        return new AndAttributeOperator(copy[0], copy[1]);
    }

    @Override
    public String toString() {
        return String.format("(%s and %s)", (Object[]) getSubconditions());
    }
}
