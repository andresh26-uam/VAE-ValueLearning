package de.hsh.inform.swa.cep.operators.attributes.comparison;

import java.util.Arrays;

import de.hsh.inform.swa.cep.Attribute;
import de.hsh.inform.swa.cep.AttributeCondition;
import de.hsh.inform.swa.cep.AttributeOperator;

/**
 * Entity Class representing a "greater than" operator (>) in the ACT.
 * @author Software Architecture Research Group
 *
 */
public class GreaterThanAttributeComparisonOperator extends AttributeOperator {

    public GreaterThanAttributeComparisonOperator(Attribute a1, Attribute a2) {
        super(a1, a2);
    }

    @Override
    public AttributeCondition copy() {
    	Attribute[] copy = Arrays.stream(getOperands()).map(cond -> cond.copy()).toArray(Attribute[]::new);
        return new GreaterThanAttributeComparisonOperator(copy[0], copy[1]);
    }

    @Override
    public String toString() {
        return String.format("(%s > %s)", (Object[]) getOperands());
    }
}
