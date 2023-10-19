package de.hsh.inform.swa.cep.operators.attributes.arithmetic;

import de.hsh.inform.swa.cep.Attribute;
import de.hsh.inform.swa.cep.AttributeOperator;
/**
 * Commonality of all arithmetic operations in the ACT.
 * So far, arithmetic operations still have the following limitations:
 * (1) only event attributes are considered
 * (2) arithmetic operations are binary and not nestable
 * @author Software Architecture Research Group
 *
 */
public abstract class ArithmeticOperator extends AttributeOperator implements Attribute{

	ArithmeticOperator(Attribute a1, Attribute a2) {
		super(a1, a2);
	}
	@Override
	public abstract ArithmeticOperator copy();
	@Override
	public abstract String toString();
	@Override
	public abstract double getMin();
	@Override
	public abstract double getMax();

	@Override
	public String getAlias() {
		return toString();
	}
}
