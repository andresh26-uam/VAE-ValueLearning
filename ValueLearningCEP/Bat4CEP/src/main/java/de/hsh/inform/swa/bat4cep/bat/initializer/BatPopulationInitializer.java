package de.hsh.inform.swa.bat4cep.bat.initializer;

import de.hsh.inform.swa.bat4cep.bat.Bat;

/**
 * Abstract class to create the swarm.
 * @author Software Architecture Research Group
 *
 */
public abstract class BatPopulationInitializer{
    private final double freqMax;
    private final double pulserate;
    private final double loudness;

    public BatPopulationInitializer(double freqMax, double pulserate, double loudness) {
        this.freqMax = freqMax;
        this.pulserate = pulserate;
        this.loudness = loudness;
    }

    public abstract Bat[] buildPopulation(int populationSize, double FREQ_MAX, double PULSERATE, double LOUDNESS, int maxEventConditionTreeHeight,
            double attributeConditionTreeRate, int maxAttributeConditionTreeHeight);

    public Bat[] buildPopulation(int populationSize, int maxEventConditionTreeHeight, double attributeConditionTreeRate, int maxAttributeConditionTreeHeight) {
        return buildPopulation(populationSize, freqMax, pulserate, loudness, maxEventConditionTreeHeight, attributeConditionTreeRate,
                maxAttributeConditionTreeHeight);
    }
}
