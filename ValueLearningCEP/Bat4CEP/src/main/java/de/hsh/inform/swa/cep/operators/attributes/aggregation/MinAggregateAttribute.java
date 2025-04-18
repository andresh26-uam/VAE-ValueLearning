package de.hsh.inform.swa.cep.operators.attributes.aggregation;

import de.hsh.inform.swa.cep.Attribute;
import de.hsh.inform.swa.cep.EventAttribute;
import de.hsh.inform.swa.cep.TemplateEvent;
/**
 * Class representing a min operation in the ACT.
 * @author Software Architecture Research Group
 *
 */
public class MinAggregateAttribute extends AggregationAttribute{
	public MinAggregateAttribute(String alias, String attributeName, TemplateEvent templateEvent) {
		super(alias, attributeName, templateEvent);
	}
	public MinAggregateAttribute(EventAttribute attr) {
		super(attr);
	}
	@Override
	public Attribute copy() {
		MinAggregateAttribute copy = new MinAggregateAttribute(getAlias(), getAttributeName(), getTemplateEvent());
		if(getWindow()!=null) copy.setWindow(getWindow());
		return copy;
	}
	@Override
	public String getAggregationFunction() {
		return "MIN";
	}
}
