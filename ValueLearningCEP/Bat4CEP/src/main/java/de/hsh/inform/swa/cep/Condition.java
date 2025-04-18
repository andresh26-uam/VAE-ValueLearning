package de.hsh.inform.swa.cep;

import java.util.Arrays;

/**
 * Commonality of all parent nodes in the ECT and ACT.
 * @author Software Architecture Research Group
 *
 */
public interface Condition {
    default int getHeight() {
        return 1 + Arrays.stream(getSubconditions()).mapToInt(Condition::getHeight).max().orElse(0);
    }

    default int getNumberOfNodes() {
        return 1 + Arrays.stream(getSubconditions()).mapToInt(Condition::getNumberOfNodes).sum();
    }

    Condition[] getSubconditions();

    void setSubcondition(Condition newOperand, int position);
}
