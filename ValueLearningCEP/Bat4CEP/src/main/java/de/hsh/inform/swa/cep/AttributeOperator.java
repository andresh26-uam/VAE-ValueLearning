package de.hsh.inform.swa.cep;

import java.util.Objects;
/**
 * Commonality of all operations in the ACT. 
 * @author Software Architecture Research Group
 *
 */
public abstract class AttributeOperator extends AttributeCondition {

    private Attribute a1, a2;
    private int conflict = 0;

    protected AttributeOperator(Attribute a1, Attribute a2) {
    	/*
    	 * A operation is a condition that cannot contain other attribute conditions as subconditions. 
    	 * Since they belong to the leaf of the ACT, the children must be elementary (constant or event attribute)
    	 */
        super(null, null); 
        this.a1 = a1;
        this.a2 = a2;
    }
    
    public void setOperand(Attribute a, int position) {
        if (position == 0) {
            a1 = a;
        } else if (position == 1) {
            a2 = a;
        }
    }

    @Override
    public int getHeight() {
        return 1;
    }

    @Override
    public int getNumberOfNodes() {
        return 1; // operator is leaf of ACT! It has operands but no subconditions.
    }

    @Override
    public AttributeCondition[] getSubconditions() {
        return null;
    }

    public Attribute[] getOperands() {
        return new Attribute[] { a1, a2 };
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        AttributeOperator that = (AttributeOperator) o;
        return Objects.equals(a1, that.a1) && Objects.equals(a2, that.a2) && this.conflict == that.conflict;
    }

    @Override
    public int hashCode() {
        return Objects.hash(a1, a2);
    }

    public void setConflict(int conflict){
        this.conflict = conflict;
    }
}
