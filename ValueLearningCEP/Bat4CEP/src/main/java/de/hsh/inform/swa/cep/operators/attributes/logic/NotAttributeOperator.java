package de.hsh.inform.swa.cep.operators.attributes.logic;

import de.hsh.inform.swa.cep.AttributeCondition;
import de.hsh.inform.swa.cep.Condition;

/**
 * Entity Class representing a "NOT" operation in the ACT.
 * @author Software Architecture Research Group
 *
 */
public class NotAttributeOperator extends AttributeCondition {

    public NotAttributeOperator(AttributeCondition cond) {
        super(cond, null);
    }

    @Override
    public AttributeCondition[] getSubconditions() {
        return new AttributeCondition[] { super.getSubconditions()[0] };
    }

    @Override
    public void setSubcondition(Condition newOperand, int position) {
        if (position == 0) super.setSubcondition(newOperand, position);
    }

    @Override
    public AttributeCondition copy() {
        return new NotAttributeOperator(getSubconditions()[0].copy());
    }

    @Override
    public String toString() {
        return String.format("(not %s)", getSubconditions()[0]);
    }
}
