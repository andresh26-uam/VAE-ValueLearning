package de.hsh.inform.swa.cep;

/**
 * Technical operator that identifies an operator with modified value range, such as the sum operator. 
 * This is required for repair purposes in the ACT, so that the algorithm is not stuck in a local maximum.
 * @author Software Architecture Research Group
 *
 */
public interface RangeOperator {
	public double getMax();
	public double getMin();
}
