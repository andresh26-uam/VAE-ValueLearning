package de.hsh.inform.swa.cep.operators.attributes.aggregation;

import de.hsh.inform.swa.cep.Attribute;
import de.hsh.inform.swa.cep.EventAttribute;
import de.hsh.inform.swa.cep.RangeOperator;
import de.hsh.inform.swa.cep.TemplateEvent;

public class SumAggregateAttribute extends AggregationAttribute implements RangeOperator{
	/**
	 * Class representing a sum operation in the ACT.
	 * @author Software Architecture Research Group
	 *
	 */
	public SumAggregateAttribute(String alias, String attributeName, TemplateEvent templateEvent) {
		super(alias, attributeName, templateEvent);
	}
	public SumAggregateAttribute(EventAttribute attr) {
		super(attr);
	}
	@Override
	public Attribute copy() {
		SumAggregateAttribute copy = new SumAggregateAttribute(getAlias(), getAttributeName(), getTemplateEvent());
		if(getWindow()!=null) copy.setWindow(getWindow());
		return copy;
	}
	@Override
	public String getAggregationFunction() {
		return "SUM";
	}
	//Since the sum operator has a larger value range than the summed attribute, the value range must be scaled accordingly.
	//For example, if the attributes of 10 events are summed, the value range of the sum is ten times larger than the summed attribute.
	@Override
	public double getMax() {
		double val = super.getMax();
		if(getWindow() != null) { 
    		return val*getWindow().getValue();
    	}
		return val;
	}
	@Override
	public double getMin() {
		double val = super.getMin();
		if(getWindow() != null) { 
    		return val*getWindow().getValue();
    	}
		return val;
	}
}
