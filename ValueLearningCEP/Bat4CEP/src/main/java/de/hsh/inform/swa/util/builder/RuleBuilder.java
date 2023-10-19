package de.hsh.inform.swa.util.builder;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import de.hsh.inform.swa.bat4cep.bat.update.EventConditionTreePointUpdate;
import de.hsh.inform.swa.cep.Action;
import de.hsh.inform.swa.cep.AttributeCondition;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.cep.EventCondition;
import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.cep.windows.Window;
import de.hsh.inform.swa.util.EventHandler;

/**
 * Class that creates a rule randomly. 
 * 
 * This includes the ECT, ACT and the window.
 * @author Software Architecture Research Group
 *
 */
public abstract class RuleBuilder {

    private final Action action;
    private final de.hsh.inform.swa.bat4cep.bat.update.EventConditionTreePointUpdate ect;
    private final WindowBuilder wbh;
    private final EventHandler eh;

    RuleBuilder(List<Event> eventTypes, WindowBuilder wbh, Action action, EventHandler eh) {
        this.wbh = wbh;
        this.action = action;
        this.eh = eh;
        this.ect = new EventConditionTreePointUpdate(eventTypes);
    }

    //delegation
    EventCondition buildRandomEventCondition() {
        return ect.getRandomEvent();
    }

    public abstract EventCondition constructConditionTree(int maxConditionTreeHeight);

    public Rule constructRule(int maxEventConditionTreeHeight, double attributeConditionTreeRate, int maxAttributeConditionTreeHeight) {
        EventCondition ect = constructConditionTree(maxEventConditionTreeHeight);
        Rule rule = new Rule(ect, buildRandomWindow(), action);
        if (maxAttributeConditionTreeHeight > 0 && ThreadLocalRandom.current().nextDouble() < attributeConditionTreeRate) {
            AttributeCondition act = AttributeConditionTreeBuilder.buildAttributeConditionTree(ect, maxAttributeConditionTreeHeight, eh);
            rule.setAttributeConditionTreeRoot(act);
        }
        return rule;
    }

    protected EventCondition buildRandomOperatorCondition(int maxHeight) {
        EventCondition left = constructConditionTree(maxHeight);
        EventCondition right;
        do { 
        	right = constructConditionTree(maxHeight);
        }while(ect.getNumberOfEventTypes()>1 && right.equals(left)); // avoid useless conditions.
        return ect.getRandomLogicalOperator(left, right);
    }

    private Window buildRandomWindow() {
        if (ThreadLocalRandom.current().nextDouble() < 0.5) {
            return wbh.getRandomLengthWindow();
        }
        return wbh.getRandomTimeWindow();
    }
}