package de.hsh.inform.swa.util.builder;

import java.time.temporal.ChronoUnit;
import java.util.concurrent.ThreadLocalRandom;

import de.hsh.inform.swa.cep.windows.LengthWindow;
import de.hsh.inform.swa.cep.windows.TimeWindow;
/**
 * Class responsible for creating random windows.
 * @author Software Architecture Research Group
 *
 */
public class WindowBuilder {

    public final long minTimeValue, maxTimeValue;
    public final int minLength, maxLength;
    private ChronoUnit timeWindowUnit;

    public WindowBuilder(int minLength, int maxLength, long minTimeValue, long maxTimeValue, ChronoUnit timeUnit) {
        this.minLength = minLength;
        this.maxLength = maxLength;
        this.minTimeValue = minTimeValue;
        this.maxTimeValue = maxTimeValue;
        this.timeWindowUnit = timeUnit;
    }

    public LengthWindow getRandomLengthWindow() {
        int value = ThreadLocalRandom.current().nextInt(minLength, maxLength);
        return new LengthWindow(value, minLength, maxLength);
    }

    public TimeWindow getRandomTimeWindow() {
        long value = ThreadLocalRandom.current().nextLong(minTimeValue, maxTimeValue);
        return new TimeWindow(value, timeWindowUnit, minTimeValue, maxTimeValue);
    }

    public int getMinLength() {
        return minLength;
    }

    public int getMaxLength() {
        return maxLength;
    }

    public long getMinTimeDifference() {
        return minTimeValue;
    }

    public long getMaxTimeDifference() {
        return maxTimeValue;
    }

    public ChronoUnit getTimeWindowUnit() {
        return timeWindowUnit;
    }
}
