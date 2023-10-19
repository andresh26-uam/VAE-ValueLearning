package de.hsh.inform.swa.cep.operators.attributes.logic;

import java.util.Arrays;

import de.hsh.inform.swa.cep.AttributeCondition;
/**
 * Entity Class representing an "OR" operation in the ACT. Currently not used in Bat4CEP.
 * @author Software Architecture Research Group
 *
 */
public class OrAttributeOperator extends AttributeCondition {

    public OrAttributeOperator(AttributeCondition cond1, AttributeCondition cond2) {
        super(cond1, cond2);
    }

    @Override
    public AttributeCondition copy() {
    	AttributeCondition[] copy = Arrays.stream(getSubconditions()).map(cond -> cond.copy()).toArray(AttributeCondition[]::new);
        return new OrAttributeOperator(copy[0], copy[1]);
    }

    @Override
    public String toString() {
        return String.format("(%s or %s)", (Object[]) getSubconditions());
    }
}
