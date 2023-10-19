package de.hsh.inform.swa.cep.operators.attributes.aggregation;

import de.hsh.inform.swa.cep.Attribute;
import de.hsh.inform.swa.cep.EventAttribute;
import de.hsh.inform.swa.cep.TemplateEvent;
/**
 * Class representing a max operation in the ACT.
 * @author Software Architecture Research Group
 *
 */
public class MaxAggregateAttribute extends AggregationAttribute{
	public MaxAggregateAttribute(String alias, String attributeName, TemplateEvent templateEvent) {
		super(alias, attributeName, templateEvent);
	}
	public MaxAggregateAttribute(EventAttribute attr) {
		super(attr);
	}
	@Override
	public Attribute copy() {
		MaxAggregateAttribute copy = new MaxAggregateAttribute(getAlias(), getAttributeName(), getTemplateEvent());
		if(getWindow()!=null) copy.setWindow(getWindow());
		return copy;
	}
	@Override
	public String getAggregationFunction() {
		return "MAX";
	}
}
