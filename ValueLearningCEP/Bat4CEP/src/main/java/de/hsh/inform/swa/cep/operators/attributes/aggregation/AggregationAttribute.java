package de.hsh.inform.swa.cep.operators.attributes.aggregation;

import de.hsh.inform.swa.cep.EventAttribute;
import de.hsh.inform.swa.cep.TemplateEvent;
import de.hsh.inform.swa.cep.windows.Window;

/**
 * Commonality of all aggregation functions in the ACT. 
 * In principle, aggregations behave like event attributes which differ only in a modified toString method and the consideration of a window
 * @author Software Architecture Research Group
 *
 */
public abstract class AggregationAttribute extends EventAttribute{
	private Window window;
	
	public AggregationAttribute(String alias, String attributeName, TemplateEvent templateEvent) {
		super(alias, attributeName, templateEvent);
	}
	public AggregationAttribute(String alias, String attributeName, TemplateEvent templateEvent, Window win) {
		this(alias, attributeName, templateEvent);
		this.window = win;
	}
	
	public AggregationAttribute(EventAttribute attr) {
		this(attr.getAlias(), attr.getAttributeName(), attr.getTemplateEvent());
	}

	public Window getWindow() {
		return window;
	}

	public void setWindow(Window window) {
		this.window = window;
	}
	@Override
	public String toString() {
		if( getTemplateEvent() != null)
			return String.format("(SELECT " + getAggregationFunction() + "(%s) FROM %s#time(%s))", getAttributeName(), getTemplateEvent().getType(), getWindow());
		return String.format("(SELECT " + getAggregationFunction() + "(%s) FROM %s#time(%s))", getAttributeName(), getAlias().charAt(0), getWindow());
    }
	
	abstract public String getAggregationFunction();
}
