package de.hsh.inform.swa.cep;

import java.util.Objects;
/**
 * Entity Class representing a constant in the ACT.
 * @author Software Architecture Research Group
 *
 */
public class ConstantAttribute implements Attribute {

    private double value;

    public ConstantAttribute(double value) {
        this.value = value;
    }

    public double getValue() {
        return value;
    }
    
    @Override
    public String getAlias() {
        return null;
    }

    @Override
    public Attribute copy() {
        return this; // immutable
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        ConstantAttribute constant = (ConstantAttribute) o;
        return Double.compare(constant.value, value) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(value);
    }

    @Override
    public String toString() {
        return Double.toString(value);
    }
    
    @Override
	public double getMin() {
		return value;
	}

	@Override
	public double getMax() {
		return value;
	}
}
