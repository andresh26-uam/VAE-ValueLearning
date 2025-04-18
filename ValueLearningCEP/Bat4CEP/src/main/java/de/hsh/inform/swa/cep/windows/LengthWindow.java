package de.hsh.inform.swa.cep.windows;

import java.util.Objects;

/**
 * Entity class representing a length window. 
 * @author Software Architecture Research Group
 *
 */
public class LengthWindow implements Window {
    public int minLength;
    public int maxLength;
    private int length;

    public LengthWindow(int length, int min, int max) {
        this.length = length;
        this.minLength = min;
        this.maxLength = max;
    }

    @Override
    public long getMaxValue() {
        return maxLength;
    }

    @Override
    public long getMinValue() {
        return minLength;
    }

    @Override
    public long getValue() {
        return length;
    }

    @Override
    public void setValue(long value) {
        this.length = Math.toIntExact(value);
    }

    @Override
    public Window copy() {
        return new LengthWindow(length, minLength, maxLength);
    }

    @Override
    public String toString() {
        return Integer.toString(length);
    }

    public int getLength() {
        return length;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        LengthWindow that = (LengthWindow) o;
        return length == that.length;
    }

    @Override
    public int hashCode() {
        return Objects.hash(length);
    }
}
