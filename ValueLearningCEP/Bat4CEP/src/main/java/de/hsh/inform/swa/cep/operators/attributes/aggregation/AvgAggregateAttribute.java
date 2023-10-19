package de.hsh.inform.swa.cep.operators.attributes.aggregation;

import de.hsh.inform.swa.cep.Attribute;
import de.hsh.inform.swa.cep.EventAttribute;
import de.hsh.inform.swa.cep.TemplateEvent;

/**
 * Entity Class representing an average operation in the ACT.
 * @author Software Architecture Research Group
 *
 */
public class AvgAggregateAttribute extends AggregationAttribute{
	public AvgAggregateAttribute(String alias, String attributeName, TemplateEvent templateEvent) {
		super(alias, attributeName, templateEvent);
	}
	public AvgAggregateAttribute(EventAttribute attr) {
		super(attr);
	}
	@Override
	public Attribute copy() {
		AvgAggregateAttribute copy = new AvgAggregateAttribute(getAlias(), getAttributeName(), getTemplateEvent());
		if(getWindow()!=null) copy.setWindow(getWindow());
		return copy;
	}
	@Override
	public String getAggregationFunction() {
		return "AVG";
	}
}
