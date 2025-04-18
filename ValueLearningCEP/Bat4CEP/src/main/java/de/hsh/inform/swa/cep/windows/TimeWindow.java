package de.hsh.inform.swa.cep.windows;

import java.time.temporal.ChronoUnit;
import java.util.Objects;

/**
 * Entity class representing a time window. 
 * @author Software Architecture Research Group
 *
 */
public class TimeWindow implements Window {

    public long minLength;
    public long maxLength;
    private long length;
    private ChronoUnit unit;

    public TimeWindow(long length, ChronoUnit unit, long minLength, long maxLength) {
        this.length = length;
        this.unit = unit;
        this.minLength = minLength;
        this.maxLength = maxLength;
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
        this.length = value;
    }

    @Override
    public Window copy() {
        return new TimeWindow(length, unit, minLength, maxLength);
    }

    @Override
    public String toString() {
        return String.format("%d sec", length);
    }

    public long getLength() {
        return length;
    }

    public ChronoUnit getUnit() {
        return unit;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TimeWindow that = (TimeWindow) o;
        return length == that.length && unit == that.unit;
    }

    @Override
    public int hashCode() {
        return Objects.hash(length, unit);
    }
}
