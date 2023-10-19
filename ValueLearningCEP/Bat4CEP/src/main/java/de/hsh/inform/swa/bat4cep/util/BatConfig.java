package de.hsh.inform.swa.bat4cep.util;

/**
 * entity class representing the hyper parameter configurations of the bat algorithm.
 * @author Software Architecture Research Group
 *
 */
public class BatConfig {
    private final String swarmName;
    private int swarmSize;
    private int timesteps;
    private double minFrequency;
    private double maxFrequency;
    private double pulserate;
    private double gamma;
    private double loudness;
    private double alpha;     
    
    public BatConfig(String swarmName, int swarmSize, int timesteps, double minFrequency, double maxFrequency, double pulserate, double gamma, double loudness, double alpha) {
    	this.swarmName = swarmName;
    	this.swarmSize  =swarmSize;
    	this.timesteps = timesteps;
    	this.minFrequency = minFrequency;
    	this.maxFrequency = maxFrequency;
    	this.pulserate = pulserate;
    	this.gamma = gamma;
    	this.loudness = loudness;
    	this.alpha = alpha;
    }
    public BatConfig copy() {
    	return new BatConfig(swarmName, swarmSize, timesteps, maxFrequency, minFrequency, pulserate, gamma, loudness, alpha);
    }
	public String getSwarmName() {
		return swarmName;
	}
	public int getSwarmSize() {
		return swarmSize;
	}
	public void setSwarmSize(int swarmSize) {
		this.swarmSize = swarmSize;
	}
	public int getTimesteps() {
		return timesteps;
	}
	public void setTimesteps(int timesteps) {
		this.timesteps = timesteps;
	}
	public double getMaxFrequency() {
		return maxFrequency;
	}
	public void setMaxFrequency(double maxFrequency) {
		this.maxFrequency = maxFrequency;
	}
	public double getPulserate() {
		return pulserate;
	}
	public void setPulserate(double pulserate) {
		this.pulserate = pulserate;
	}
	public double getGamma() {
		return gamma;
	}
	public void setGamma(double gamma) {
		this.gamma = gamma;
	}
	public double getLoudness() {
		return loudness;
	}
	public void setLoudness(double loudness) {
		this.loudness = loudness;
	}
	public double getAlpha() {
		return alpha;
	}
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
	public double getMinFrequency() {
		return minFrequency;
	}
	public void setMinFrequency(double minFrequency) {
		this.minFrequency = minFrequency;
	}
}
