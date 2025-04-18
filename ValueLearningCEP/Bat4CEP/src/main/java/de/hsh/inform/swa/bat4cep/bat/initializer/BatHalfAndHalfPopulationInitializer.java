package de.hsh.inform.swa.bat4cep.bat.initializer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import de.hsh.inform.swa.bat4cep.bat.Bat;
import de.hsh.inform.swa.cep.Action;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.util.EventHandler;
import de.hsh.inform.swa.util.builder.WindowBuilder;

/**
 * Class that generates the swarm.
 * Here, half of the swarm is filled with bats that exploit the maximum ECT and ACT heights
 * and the other half is filled with bats that do not utilize these limits (i.e., ACT and ECT are shorter than the maximum limits).
 * @author Software Architecture Research Group
 *
 */
public class BatHalfAndHalfPopulationInitializer extends BatPopulationInitializer {
    private BatFullMethodPopulationInitializer full;
    private BatGrowMethodPopulationInitializer grow;

    public BatHalfAndHalfPopulationInitializer(double FREQ_MAX, double PULSERATE, double LOUDNESS, List<Event> eventTypes, WindowBuilder wbh, Action action,
            EventHandler eh) {
        super(FREQ_MAX, PULSERATE, LOUDNESS);
        full = new BatFullMethodPopulationInitializer(FREQ_MAX, PULSERATE, LOUDNESS, eventTypes, wbh, action, eh);
        grow = new BatGrowMethodPopulationInitializer(FREQ_MAX, PULSERATE, LOUDNESS, eventTypes, wbh, action, eh);
    }

    @Override
    public Bat[] buildPopulation(int populationSize, double FREQ_MAX, double PULSERATE, double LOUDNESS, int maxEventConditionTreeHeight,
            double attributeConditionTreeRate, int maxAttributeConditionTreeHeight) {
        List<Bat> result = new ArrayList<>();

        int firstSize = populationSize / 2;
        int secondSize = populationSize - firstSize;
        result.addAll(Arrays.asList(
                full.buildPopulation(firstSize, FREQ_MAX, PULSERATE, LOUDNESS, maxEventConditionTreeHeight, attributeConditionTreeRate,
                        maxAttributeConditionTreeHeight)));
        result.addAll(Arrays.asList(
                grow.buildPopulation(secondSize, FREQ_MAX, PULSERATE, LOUDNESS, maxEventConditionTreeHeight, attributeConditionTreeRate,
                        maxAttributeConditionTreeHeight)));

        
        return result.toArray(new Bat[result.size()]);
    }

}
