package de.hsh.inform.swa.cep;
/**
 * Commonality of all parent nodes in the ACT. This includes logical operation like AND or NOT and comparisons like <, >, =.
 * @author Software Architecture Research Group
 *
 */
public abstract class AttributeCondition implements Condition {

    private AttributeCondition cond1, cond2;

    public AttributeCondition(AttributeCondition cond1, AttributeCondition cond2) {
        this.cond1 = cond1;
        this.cond2 = cond2;
    }

    abstract public AttributeCondition copy();

    @Override
    public void setSubcondition(Condition newOperand, int position) {
        if (newOperand instanceof AttributeCondition) {
            if (position == 0) {                
                cond1 = (AttributeCondition) newOperand;
            } else if (position == 1) {
                cond2 = (AttributeCondition) newOperand;
            }
        }
    }

    @Override
    public AttributeCondition[] getSubconditions() {
        return new AttributeCondition[] { cond1, cond2 };
    }

    @Override
    public boolean equals(Object o) {
        AttributeCondition c = (AttributeCondition) o;
        if (this == null || c == null) {
            return false;
        }
        return this.cond1 != null && this.cond2 != null && c.cond1 != null && c.cond2 != null && this.cond1.equals(c.cond1) && this.cond2.equals(c.cond2);
    }
}
