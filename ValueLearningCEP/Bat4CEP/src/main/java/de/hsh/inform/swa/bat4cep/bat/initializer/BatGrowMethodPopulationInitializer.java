package de.hsh.inform.swa.bat4cep.bat.initializer;

import java.util.List;

import de.hsh.inform.swa.cep.Action;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.util.EventHandler;
import de.hsh.inform.swa.util.builder.GrowRuleBuilder;
import de.hsh.inform.swa.util.builder.RuleBuilder;
import de.hsh.inform.swa.util.builder.WindowBuilder;
/**
 * Class that generates a part of the swarm with semi-finished bat positions (i.e. the maximum ACT and ECT heights are not fully utilized).
 * @author Software Architecture Research Group
 *
 */
public class BatGrowMethodPopulationInitializer extends RuleBuilderPopulationInitializer {
    public BatGrowMethodPopulationInitializer(double freqMax, double pulserate, double loudness, List<Event> eventTypes, WindowBuilder wbh, Action action,
            EventHandler eh) {
        super(freqMax, pulserate, loudness, eventTypes, wbh, action, eh);
    }   

    @Override
    public RuleBuilder buildRuleBuilder(List<Event> eventTypes, WindowBuilder wbh, Action action, EventHandler eh) {
        return new GrowRuleBuilder(eventTypes, wbh, action, eh);
    }
}
