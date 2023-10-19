package de.hsh.inform.swa.cep.operators.attributes.arithmetic;

import java.util.Arrays;

import de.hsh.inform.swa.cep.Attribute;
import de.hsh.inform.swa.cep.RangeOperator;
/**
 * Entity Class representing an addition operation in the ACT.
 * @author Software Architecture Research Group
 *
 */
public class AdditionOperator extends ArithmeticOperator implements RangeOperator{

	public AdditionOperator(Attribute a1, Attribute a2) {
		super(a1, a2);
	}
	@Override
    public AdditionOperator copy() {
    	Attribute[] copy = Arrays.stream(getOperands()).map(cond -> cond.copy()).toArray(Attribute[]::new);
        return new AdditionOperator(copy[0], copy[1]);
    }
	@Override
    public String toString() {
        return String.format("%s + %s", (Object[]) getOperands());
    }
	@Override
	public double getMin() {
		return Arrays.stream(getOperands()).mapToDouble(op -> op.getMin()).min().getAsDouble()*2;
	}
	@Override
	public double getMax() {
		return Arrays.stream(getOperands()).mapToDouble(op -> op.getMax()).max().getAsDouble()*2;
	}
}
