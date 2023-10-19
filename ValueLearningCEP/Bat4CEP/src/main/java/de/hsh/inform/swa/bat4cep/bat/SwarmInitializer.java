package de.hsh.inform.swa.bat4cep.bat;

import java.util.Arrays;

import de.hsh.inform.swa.bat4cep.bat.initializer.BatPopulationInitializer;
import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.evaluation.RuleEvaluator;
import de.hsh.inform.swa.evaluation.RuleWithFitness;
import de.hsh.inform.swa.util.FitnessHelper;
/**
 * Delegation class that creates and evaluates a swarm.
 * @author Software Architecture Research Group
 *
 */
public class SwarmInitializer {
	private static final double ACT_RATE = 1.0;	//an act rate of 1 means that each rule should have an act tree
    public static Bat[] initSwarm(int currentSwarmSize, boolean initializeOnlyOnce, BatPopulationInitializer initializer, RuleEvaluator evaluator,
            int maxECTheight, int maxACTheight) {
    	Bat[] swarm = initializer.buildPopulation(currentSwarmSize, maxECTheight, ACT_RATE, maxACTheight);
    	//measure fitness
    	RuleWithFitness[] rules = Arrays.stream(swarm).map(b -> (Rule) b.getSolution()).toArray(size -> new RuleWithFitness[size]);
        FitnessHelper.measureFitness(evaluator, rules);
        return swarm;
    }
}