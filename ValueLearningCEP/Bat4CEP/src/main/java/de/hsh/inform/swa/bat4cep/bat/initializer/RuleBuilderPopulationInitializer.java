package de.hsh.inform.swa.bat4cep.bat.initializer;

import java.util.List;

import de.hsh.inform.swa.bat4cep.bat.Bat;
import de.hsh.inform.swa.bat4cep.bat.BatUtils;
import de.hsh.inform.swa.cep.Action;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.evaluation.RuleWithFitness;
import de.hsh.inform.swa.util.EventHandler;
import de.hsh.inform.swa.util.builder.RuleBuilder;
import de.hsh.inform.swa.util.builder.WindowBuilder;

/**
 * Abstract class for all types of population initialization.
 * @author Software Architecture Research Group
 *
 */
public abstract class RuleBuilderPopulationInitializer extends BatPopulationInitializer {

    private final RuleBuilder ruleBuilder;

	public RuleBuilderPopulationInitializer(double freqMax, double pulserate, double loudness, List<Event> eventTypes, WindowBuilder wbh, Action action,
            EventHandler eh) {
        super(freqMax, pulserate, loudness);
        this.ruleBuilder = buildRuleBuilder(eventTypes, wbh, action, eh);
    }

    abstract public RuleBuilder buildRuleBuilder(List<Event> eventTypes, WindowBuilder wbh, Action action, EventHandler eh);

    @Override
    public Bat[] buildPopulation(int populationSize, double FREQ_MAX, double PULSERATE, double LOUDNESS, int maxEventConditionTreeHeight,
            double attributeConditionTreeRate, int maxAttributeConditionTreeHeight) {
        Bat[] population = new Bat[populationSize];
        for (int i = 0; i < population.length; i++) {
            RuleWithFitness solution = new RuleWithFitness(
                    getRuleBuilder().constructRule(maxEventConditionTreeHeight, attributeConditionTreeRate, maxAttributeConditionTreeHeight));
            population[i] = BatUtils.initializeBat(FREQ_MAX, PULSERATE, LOUDNESS, solution);
        }
        return population;
    }
    
    public RuleBuilder getRuleBuilder() {
		return ruleBuilder;
	}
    
}
