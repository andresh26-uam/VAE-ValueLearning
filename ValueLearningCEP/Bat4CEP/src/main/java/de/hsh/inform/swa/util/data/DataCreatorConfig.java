package de.hsh.inform.swa.util.data;

import java.util.HashMap;
import java.util.Map;

import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.cep.Rule;
/**
 * Simple entity class with getters and setters.
 * Contains all essential informations about the data set.
 * @author Software Architecture Research Group
 *
 */
public class DataCreatorConfig {
    private Rule rule;
    private int numEvents;
    private int numEventTypes;
    private int numEventAttributes;
    private int minIntermediateSeconds;
    private int maxIntermediateSeconds;
    private double hitPercentage;
    private AttributeConfig<?> defaultRange;
    private final Map<String, AttributeConfig<?>> fixedSensorRange = new HashMap<>();
    private Event complexEvent;
    private DatasetEnum data;

    public DataCreatorConfig copy() {
        DataCreatorConfig copy = new DataCreatorConfig();
        copy.setRule(rule);
        copy.setNumEvents(numEvents);
        copy.setNumEventTypes(numEventTypes);
        copy.setNumEventAttributes(numEventAttributes);
        copy.setMinIntermediateSeconds(minIntermediateSeconds);
        copy.setMaxIntermediateSeconds(maxIntermediateSeconds);
        copy.getFixedSensorRange().putAll(fixedSensorRange);
        copy.setDefaultRange(defaultRange);
        copy.setComplexEvent(complexEvent);
        copy.setHitPercentage(hitPercentage);
        copy.setData(data);
        return copy;
    }

	public Rule getRule() {
		return rule;
	}

	public void setRule(Rule rule) {
		this.rule = rule;
	}

	public int getNumEvents() {
		return numEvents;
	}

	public void setNumEvents(int numEvents) {
		this.numEvents = numEvents;
	}

	public int getNumEventTypes() {
		return numEventTypes;
	}

	public void setNumEventTypes(int numEventTypes) {
		this.numEventTypes = numEventTypes;
	}

	public int getNumEventAttributes() {
		return numEventAttributes;
	}

	public void setNumEventAttributes(int numEventAttributes) {
		this.numEventAttributes = numEventAttributes;
	}

	public int getMinIntermediateSeconds() {
		return minIntermediateSeconds;
	}

	public void setMinIntermediateSeconds(int minIntermediateSeconds) {
		this.minIntermediateSeconds = minIntermediateSeconds;
	}

	public int getMaxIntermediateSeconds() {
		return maxIntermediateSeconds;
	}

	public void setMaxIntermediateSeconds(int maxIntermediateSeconds) {
		this.maxIntermediateSeconds = maxIntermediateSeconds;
	}

	public AttributeConfig<?> getDefaultRange() {
		return defaultRange;
	}

	public void setDefaultRange(AttributeConfig<?> defaultRange) {
		this.defaultRange = defaultRange;
	}

	public Event getComplexEvent() {
		return complexEvent;
	}

	public void setComplexEvent(Event complexEvent) {
		this.complexEvent = complexEvent;
	}

	public DatasetEnum getData() {
		return data;
	}

	public void setData(DatasetEnum data) {
		this.data = data;
	}

	public Map<String, AttributeConfig<?>> getFixedSensorRange() {
		return fixedSensorRange;
	}

	public double getHitPercentage() {
		return hitPercentage;
	}

	public void setHitPercentage(double hitPercentage) {
		this.hitPercentage = hitPercentage;
	}
}
