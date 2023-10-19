package de.hsh.inform.swa.util;

import java.time.temporal.ChronoUnit;
import java.util.Date;
import java.util.List;

import de.hsh.inform.swa.cep.Event;
/**
 * Helper class that determines the minimum and maximum time distance in the entire event stream.
 * @author Software Architecture Research Group
 *
 */
public class TimeUtils {
    public static long getMaximumTimeDistance(List<Event> events, ChronoUnit unit) {
        if (events.isEmpty()) return 0;
        Date timeOfFirstEvent = events.get(0).getTime();
        Date timeOfLastEvent = events.get(events.size() - 1).getTime();
        
        return timeOfFirstEvent.toInstant().until(timeOfLastEvent.toInstant(), unit);
    }

    public static long getMinimumTimeDistance(List<Event> events, ChronoUnit unit) {
        if (events.isEmpty()) return 0;
        long minTimeDistance = events.get(0).getTime().toInstant().until(events.get(1).getTime().toInstant(), unit);
        for (int i = 2; i < events.size(); i++) {
            long currentDistance = events.get(i - 1).getTime().toInstant().until(events.get(i).getTime().toInstant(), unit);
            if (currentDistance < minTimeDistance) {
                minTimeDistance = currentDistance;
            }
        }
        return minTimeDistance;
    }

    public static final String formatTime(long millis) {
        long secs = millis / 1000;
        return String.format("%02d:%02d:%02d", secs / 3600, (secs % 3600) / 60, secs % 60);
    }
}
