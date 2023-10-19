package de.hsh.inform.swa.cep;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Class containing metadata of a specific event type.
 * @author Software Architecture Research Group
 *
 */
public class TemplateEvent implements Comparable<TemplateEvent> {
    private final String type;
    private final Map<String, Double> attributeMaximums = new HashMap<>(), attributeMinimums = new HashMap<>();
    private final Set<String> attributes = new HashSet<>();

    public TemplateEvent(Event event) {
        this.type = event.getType();
        this.attributes.addAll(event.getAttributeNames());
        updateExtremes(event);
    }

    public void updateExtremes(Event event) {
        for (String attribute : getAttributes()) {
            double eventValue = Double.parseDouble(event.getValue(attribute).toString());
            Double currentMax = getAttributeMaximums().get(attribute);
            if (currentMax == null || currentMax < eventValue) {
                getAttributeMaximums().put(attribute, eventValue);
            }
            Double currentMin = getAttributeMinimums().get(attribute);
            if (currentMin == null || currentMin > eventValue) {
                getAttributeMinimums().put(attribute, eventValue);
            }
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getType()).append(" (");
        for (String attribute : getAttributes()) {
            sb.append(attribute).append(":").append(getAttributeMinimums().get(attribute)).append(", ").append(getAttributeMaximums().get(attribute)).append(" ");
        }
        return sb.append(")").toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TemplateEvent that = (TemplateEvent) o;
        return Objects.equals(getType(), that.getType()) && Objects.equals(getAttributeMaximums(), that.getAttributeMaximums())
                && Objects.equals(getAttributeMinimums(), that.getAttributeMinimums()) && Objects.equals(getAttributes(), that.getAttributes());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getType(), getAttributeMaximums(), getAttributeMinimums(), getAttributes());
    }

    @Override
    public int compareTo(TemplateEvent o) {
        return getType().compareTo(o.getType());
    }

	public Map<String, Double> getAttributeMinimums() {
		return attributeMinimums;
	}

	public Map<String, Double> getAttributeMaximums() {
		return attributeMaximums;
	}

	public Set<String> getAttributes() {
		return attributes;
	}

	public String getType() {
		return type;
	}
}