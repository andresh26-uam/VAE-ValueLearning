package de.hsh.inform.swa.bat4cep;

import de.hsh.inform.swa.bat4cep.util.BatConfig;
import de.hsh.inform.swa.util.data.DataCreatorConfig;
/**
 * Class representing a test case. 
 * @author Software Architecture Research Group 
 *
 */
public class BatTestCase {
	private final DataCreatorConfig trainingData;	
	private final BatConfig batConfig;
	private final int numRuns;
	private int maxECTHeight = 3;
	private int maxACTHeight = 3;
	
	public BatTestCase(DataCreatorConfig trainingData, BatConfig batConfig, int numRuns) {
		super();
		this.trainingData = trainingData;
		this.batConfig = batConfig;
		this.numRuns = numRuns;
		
		// Define maximum size for the start population. Since in a real use case the rule is not known in advance,
		// the following two lines can be removed and an appropriate size for the application domain should be selected.
		// If you have no idea for a correct size, you should set the heights large. 
		// maxECTHeight = 3 and maxACTHeight = 3 should be sufficient for most real world scenarios.
		this.maxECTHeight = trainingData.getRule().getEventConditionTreeRoot().getHeight();
		this.maxACTHeight = trainingData.getRule().getAttributeConditionTreeRoot().getHeight();
	}

	public int getMaxECTHeight() {
		return maxECTHeight;
	}

	public void setMaxECTHeight(int maxECTHeight) {
		this.maxECTHeight = maxECTHeight;
	}

	public int getMaxACTHeight() {
		return maxACTHeight;
	}

	public void setMaxACTHeight(int maxACTHeight) {
		this.maxACTHeight = maxACTHeight;
	}

	public DataCreatorConfig getTrainingData() {
		return trainingData;
	}

	public BatConfig getBatConfig() {
		return batConfig;
	}

	public int getNumRuns() {
		return numRuns;
	}

}
