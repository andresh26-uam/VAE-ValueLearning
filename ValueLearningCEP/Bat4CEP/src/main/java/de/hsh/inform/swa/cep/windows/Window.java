package de.hsh.inform.swa.cep.windows;

/**
 * Commonality of all possible windows for a rule. 
 * @author Software Architecture Research Group
 *
 */
public interface Window {
    long getMaxValue();

    long getMinValue();

    long getValue();

    void setValue(long value);

    Window copy();
}
