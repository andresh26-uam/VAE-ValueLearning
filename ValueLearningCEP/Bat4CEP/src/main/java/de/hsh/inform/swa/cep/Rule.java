package de.hsh.inform.swa.cep;

import java.util.HashSet;
import de.hsh.inform.swa.cep.windows.Window;

/**
 * Entity class representing a rule, which consists of ECT, ACT, Window and the complex event.
 * @author Software Architecture Research Group
 *
 */
public class Rule {

    private Window window;
    private final Action action;
    
    private EventCondition eventConditionTreeRoot;
    private AttributeCondition attributeConditionTreeRoot;
    
    public Rule(EventCondition c, Window w, Action a) {
    	this.eventConditionTreeRoot = c;
    	this.window = w;
    	this.action = a;
    }

	public AttributeCondition getAttributeConditionTreeRoot() {
        return attributeConditionTreeRoot;
    }

    public void setAttributeConditionTreeRoot(AttributeCondition attributeConditionTreeRoot) {
        this.attributeConditionTreeRoot = attributeConditionTreeRoot;
    }

    public EventCondition getEventConditionTreeRoot() {
        return eventConditionTreeRoot;
    }

    public void setEventConditionTreeRoot(EventCondition eventConditionTreeRoot) {
        this.eventConditionTreeRoot = eventConditionTreeRoot;
    }

    public Window getWindow() {
        return window;
    }

    public void setWindow(Window window) {
        this.window = window;
    }
    
    public Action getAction() {
        return action;
    }
    
    @Override
    public String toString() {
    	if (getAttributeConditionTreeRoot() == null) {
            return getPatternAsString();
        }
        return String.format("[%s] where %s", getPatternAsString(), getAttributeConditionTreeRoot());
    }

    @Override
    public boolean equals(Object o) {
        return this.toString().equals(((Rule) o).toString());
    }

    public String getPatternAsString() {
    	if(getWindow() != null) {
    		return String.format("every(%s) where timer:within(%s)", getEventConditionTreeRoot().toStringWithAlias(new HashSet<>()), getWindow());
    	}
    	return String.format("every(%s)", getEventConditionTreeRoot().toStringWithAlias(new HashSet<>()));
    }

    public Rule copy() {
        Rule ruleCopy = new Rule(getEventConditionTreeRoot().copy(), getWindow().copy(), getAction().copy());
        if (getAttributeConditionTreeRoot() != null) {
            ruleCopy.setAttributeConditionTreeRoot(getAttributeConditionTreeRoot().copy());
        }
        return ruleCopy;
    }
}
