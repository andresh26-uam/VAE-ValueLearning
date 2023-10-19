package de.hsh.inform.swa.cep;

import java.util.Objects;

/**
 * Entity Class representing a event attribute in the ACT.
 * @author Software Architecture Research Group
 *
 */
public class EventAttribute implements Attribute {

    private final String alias, attributeName;
    private final TemplateEvent templateEvent;
    
    public EventAttribute(String alias, String attributeName, TemplateEvent templateEvent) {
        this.alias = alias;
        this.attributeName = attributeName;
        this.templateEvent = templateEvent;
    }

    @Override
    public Attribute copy() {
        return this; // immutable
    }

    @Override
    public String getAlias() {
        return alias;
    }

    public String getAttributeName() {
        return attributeName;
    }

    public TemplateEvent getTemplateEvent() {
        return templateEvent;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        EventAttribute eventAttribute = (EventAttribute) o;
        return Objects.equals(alias, eventAttribute.alias) && Objects.equals(attributeName, eventAttribute.attributeName);
    }

    @Override
    public int hashCode() {
        return Objects.hash(alias, attributeName);
    }

    @Override
    public String toString() {
        return String.format("%s.%s", alias, attributeName);
    }
    @Override
	public double getMin() {
		return getTemplateEvent().getAttributeMinimums().get(getAttributeName());
	}

	@Override
	public double getMax() {
		return getTemplateEvent().getAttributeMaximums().get(getAttributeName());
	}
}
