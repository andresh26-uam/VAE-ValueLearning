package de.hsh.inform.swa.util.data;

import java.util.concurrent.ThreadLocalRandom;
/**
 * Entity class that contains the minimum and maximum possible value of an attribute type.
 * @author Software Architecture Research Group
 *
 * @param <T> mainly Double or Integer
 */
public class AttributeConfig<T extends Number> {
    private final T minValue;
    private final T maxValue;
    private final Class<? extends Number> type;
    
    public AttributeConfig(T min, T max) {
        this.minValue = min;
        this.maxValue = max;
        this.type = min.getClass();
    }

    public Number getRandomValue() {
        if (type.equals(Integer.class)) {
        	return ThreadLocalRandom.current().nextInt(maxValue.intValue() - minValue.intValue()+1) + minValue.intValue();
        }
        return ((int)((ThreadLocalRandom.current().nextDouble() * (maxValue.longValue() - minValue.longValue()+1) + minValue.doubleValue())*100))/100.0;

    }

	public T getMinValue() {
		return minValue;
	}

	public T getMaxValue() {
		return maxValue;
	}

	public Class<? extends Number> getType() {
		return type;
	}
}
