package de.hsh.inform.swa.cep;

import java.text.SimpleDateFormat;
import java.time.Instant;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
/**
 * Entity class representing events in the ECT.
 * @author Software Architecture Research Group
 *
 */
public class Event extends EventCondition implements Comparable<Event> {
    private final String type;
    private static final SimpleDateFormat TIME_CONVERTER = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

    private int number;
    private Date time;
    private Map<String, Object> attributes = Collections.unmodifiableMap(new HashMap<>());

    public Event(String type) {
    	this(type, Date.from(Instant.now()));
    }


    public Event(String type, Date time2, Map<String, Object> attributes) {
        this(type, time2);
        this.attributes = Collections.unmodifiableMap(attributes);
    }


    public Event(String type, Date time) {
    	super(null, null);
        this.type = type;
        this.time = time;

    }

    public Map<String, Object> getAttributes() {
        Map<String, Object> attributePlusLineNumber = new HashMap<>(attributes);
        attributePlusLineNumber.put("_lineNumber", this.number);
        return attributePlusLineNumber;
    }

    public Object getValue(String attribute) {
        return this.attributes.get(attribute);
    }

    public Set<String> getAttributeNames() {
        return attributes.keySet();
    }

    @Override
    public int getHeight() {
        return 0;
    }

    @Override
    public EventCondition[] getSubconditions() {
        return null;
    }

    @Override
    public void setSubcondition(Condition newOperand, int position) {}

    @Override
    public EventCondition copy() {
        return this; // immutable
    }

    @Override
    public int getNumberOfNodes() {
        return 1; //leaf
    }

    @Override
    public String toString() {
        return (time == null ? "" : TIME_CONVERTER.format(time) + "; ") + type;
    }

    @Override
    public String toStringWithAlias(Set<String> aliasesSoFar) {
        int number = 0;
        String alias = type + number;
        while (aliasesSoFar.contains(alias)) {
            number++;
            alias = type + number;
        }
        aliasesSoFar.add(alias);
        return alias + " = " + type;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        Event event = (Event) o;
        return Objects.equals(type, event.type) && Objects.equals(time, event.time) && Objects.equals(attributes, event.attributes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, time, attributes);
    }

    public long getTimestamp() {
        return time.toInstant().toEpochMilli();
    }

    public String getType() {
        return type;
    }

    @Override
    public int compareTo(Event o) {
        return time.compareTo(o.time);
    }


	public Date getTime() {
		return time;
	}


	public int getNumber() {
		return number;
	}


	public void setNumber(int number) {
		this.number = number;
	}


	public void setTime(Date time) {
		this.time = time;
	}
}
