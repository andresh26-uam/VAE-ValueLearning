package de.hsh.inform.swa.util.builder;

import java.util.List;

import de.hsh.inform.swa.cep.Action;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.cep.EventCondition;
import de.hsh.inform.swa.util.EventHandler;
/**
 * Class to create the ECT. 
 * 
 * All ECTS created with this class have the maximum defined ECT length.
 * @author Software Architecture Research Group
 *
 */
public class FullRuleBuilder extends RuleBuilder {

    public FullRuleBuilder(List<Event> eventTypes, WindowBuilder wbh, Action action, EventHandler eh) {
        super(eventTypes, wbh, action, eh);
    }

    @Override
    public EventCondition constructConditionTree(int maxHeight) {
        if (maxHeight == 0) {
            return buildRandomEventCondition();
        }
        return buildRandomOperatorCondition(maxHeight - 1);
    }
}
