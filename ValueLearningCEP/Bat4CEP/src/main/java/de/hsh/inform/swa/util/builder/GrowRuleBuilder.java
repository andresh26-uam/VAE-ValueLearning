package de.hsh.inform.swa.util.builder;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import de.hsh.inform.swa.cep.Action;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.cep.EventCondition;
import de.hsh.inform.swa.util.EventHandler;
/**
 * Class to create the ECT. 
 * 
 * All ECTs created with this class have a smaller length than the maximum defined ECT length.
 * @author Software Architecture Research Group
 *
 */
public class GrowRuleBuilder extends RuleBuilder {

    public GrowRuleBuilder(List<Event> eventTypes, WindowBuilder wbh, Action action, EventHandler eh) {
        super(eventTypes, wbh, action, eh);
    }

    @Override
    public EventCondition constructConditionTree(int maxHeight) {
        double eventChance = ThreadLocalRandom.current().nextInt(maxHeight + 1);
        if (eventChance == 0) {
            return buildRandomEventCondition();
        }
        return buildRandomOperatorCondition(maxHeight - 1);
    }
}
