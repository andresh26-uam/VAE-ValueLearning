package de.hsh.inform.swa.evaluation.esper;

import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;
import java.util.concurrent.ConcurrentSkipListSet;

import com.espertech.esper.event.map.MapEventBean;
/**
 * Subscriber class that receives all matches from the engine and determines the number of hits. 
 * @author Software Architecture Research Group
 *
 */
public class EsperSubscriber {
	private Set<Integer> hits = new ConcurrentSkipListSet<Integer>();

    public EsperSubscriber() {}

    public void update(Map<String, MapEventBean> events) {
        // The argument is a map of all events that match the subscribed rule. We need to grab the latest one.
        // Thats the one with the highest line number
        OptionalInt highestCount = events.values().stream().mapToInt(bean -> (Integer) bean.getProperties().get("_lineNumber")).max();
        if (highestCount.isPresent()) {
            hits.add(highestCount.getAsInt() + getOffset());
        }
    }

    public int getOffset() { //needed for inheritance purposes
    	return 0;
    }
    public Set<Integer> getFiredPosition() {
        return hits;
    }

}
