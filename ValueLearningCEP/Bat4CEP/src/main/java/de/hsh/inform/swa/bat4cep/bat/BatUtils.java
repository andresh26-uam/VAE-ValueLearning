package de.hsh.inform.swa.bat4cep.bat;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import de.hsh.inform.swa.evaluation.RuleWithFitness;
/**
 * Helper class to reduce the code in BatAlgorithm.java
 * @author Software Architecture Research Group
 *
 */
public class BatUtils {

    public static double getAvgLoudness(Bat[] swarm) {
        return Arrays.stream(swarm).mapToDouble(b -> b.getA()).average().getAsDouble();
    }

    public static double getAvgPulserate(Bat[] swarm) {
        return Arrays.stream(swarm).mapToDouble(b -> b.getR()).average().getAsDouble();

    }

    public static double getAvgFitness(Bat[] swarm) {
        return Arrays.stream(swarm).mapToDouble(b -> b.getSolution().getTotalFitness()).average().getAsDouble();
    }

    public static Bat initializeBat(double FREQ_MAX, double PULSERATE, double LOUDNESS, RuleWithFitness sol) {
        Bat bat = new Bat();
        bat.setF(ThreadLocalRandom.current().nextDouble(FREQ_MAX));
        bat.setV(ThreadLocalRandom.current().nextDouble(bat.getF()));
        bat.setR(PULSERATE * ThreadLocalRandom.current().nextDouble(0.85, 1.0));
        bat.setA(LOUDNESS * ThreadLocalRandom.current().nextDouble(0.85, 1.0));

        bat.setSolution(sol);
        return bat;

    }
}
