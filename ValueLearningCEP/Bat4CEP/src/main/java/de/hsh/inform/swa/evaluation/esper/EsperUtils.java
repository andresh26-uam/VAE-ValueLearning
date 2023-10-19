 package de.hsh.inform.swa.evaluation.esper;

import com.espertech.esper.client.EPAdministrator;
import com.espertech.esper.client.EPStatement;

import de.hsh.inform.swa.cep.Rule;

/**
 * Helper class that compiles a rule to an Esper EPL statement.
 * @author Software Architecture Research Group
 */
public class EsperUtils {
    public static EPStatement createStatement(EPAdministrator administrator, Rule rule){
        if(rule.getAttributeConditionTreeRoot() == null) {
            return administrator.createEPL(String.format("select * from pattern [%s]", rule.getPatternAsString()));
        }
        return administrator.createEPL(String.format("select * from pattern [%s] where %s", rule.getPatternAsString(), rule.getAttributeConditionTreeRoot()));
    }
}
