package de.hsh.inform.swa.cep;
/**
 * Commonality of all leaves in the ACT. This includes event attributes, aggregations and constants.
 * @author Software Architecture Research Group
 *
 */
public interface Attribute {
    String getAlias();
    Attribute copy();
    
    double getMin();
    double getMax();
    
}
