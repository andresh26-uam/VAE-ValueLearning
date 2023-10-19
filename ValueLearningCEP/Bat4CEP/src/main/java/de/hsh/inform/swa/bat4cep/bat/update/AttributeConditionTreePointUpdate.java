package de.hsh.inform.swa.bat4cep.bat.update;

import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import de.hsh.inform.swa.cep.Attribute;
import de.hsh.inform.swa.cep.AttributeCondition;
import de.hsh.inform.swa.cep.AttributeOperator;
import de.hsh.inform.swa.cep.RangeOperator;
import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.cep.TemplateEvent;
import de.hsh.inform.swa.util.EventHandler;
import de.hsh.inform.swa.util.builder.AttributeConditionTreeBuilder;
import de.hsh.inform.swa.util.builder.AttributeConditionTreeOperators;
import de.hsh.inform.swa.util.builder.ConditionTreeTraverser;
/**
 * Class responsible for updates to the ACT of a rule.
 * These updates include changes to ...
 * ... operators (and, ->, or, +, -, ...)
 * ... event attributes
 * ... constants.
 * Works closely with AttributeConditionTreeBuilder.java.
 * 
 * @author Software Architecture Research Group
 */
public class AttributeConditionTreePointUpdate {

    public static void update(Rule rule, EventHandler eh, int mutationPoint) {
    	
    	AttributeCondition act = rule.getAttributeConditionTreeRoot();
        AttributeCondition originalAttributeCondition = (AttributeCondition) ConditionTreeTraverser.getConditionWithPreOrderIndex(act, mutationPoint);
        Map<TemplateEvent, Integer> occurrences = ConditionTreeTraverser.getTemplateEventsOccurrencesOfEct(rule.getEventConditionTreeRoot(), eh);
        
        if (originalAttributeCondition == null) {
            rule.setAttributeConditionTreeRoot(AttributeConditionTreeBuilder.buildRandomAttributeComparisonOperator(occurrences));
        } else if (originalAttributeCondition instanceof AttributeOperator) {
            AttributeOperator op = (AttributeOperator) originalAttributeCondition;
            Attribute[] operands = op.getOperands();
            int subMutationPoint = ThreadLocalRandom.current().nextInt(1 + operands.length); // operation itself + operands
            if (subMutationPoint == 0) {
                rule.setAttributeConditionTreeRoot(ConditionTreeTraverser.replaceNode(act, updateComparisonOperator(op), mutationPoint));
            } else {
                subMutationPoint -= 1;
                updateOperandOf(op, subMutationPoint, occurrences);
            }
        } else {
            rule.setAttributeConditionTreeRoot(
                    ConditionTreeTraverser.replaceNode(act, getNewAttributeOperator(originalAttributeCondition, occurrences, rule, eh), mutationPoint));
        }
    }

    private static AttributeOperator updateComparisonOperator(AttributeOperator op) {
        Attribute[] operands = op.getOperands();
        return AttributeConditionTreeOperators.getRandomComparisonOperator(operands[0], operands[1]);
    }

    private static void updateOperandOf(AttributeOperator op, int operandNumber, Map<TemplateEvent, Integer> occurrences) {
    	Attribute oldFirst, first;
        oldFirst = op.getOperands()[0];
        if (operandNumber == 0) {
        	first = AttributeConditionTreeOperators.getRandomEventAttributeOfEct(occurrences);
            op.setOperand(first, 0);
            //some operators require special treatment. If you are a member of the
            //Software Architecture research group, please consider the additional technical documentation
            if(first instanceof RangeOperator || oldFirst instanceof RangeOperator) {
            	op.setOperand(AttributeConditionTreeBuilder.buildRandomSecondAttribute(first, occurrences), 1);
            }
        } else { // operandNumber == 1
            Attribute newSecond = AttributeConditionTreeBuilder.buildRandomSecondAttribute(oldFirst, occurrences);
            op.setOperand(newSecond, 1);
        }
    }

    private static AttributeCondition getNewAttributeOperator(AttributeCondition originalAttributeCondition, Map<TemplateEvent, Integer> occurrences, Rule rule,
            EventHandler eh) {
        
    	AttributeCondition[] children = originalAttributeCondition.getSubconditions();
        AttributeCondition first = children[0];
        AttributeCondition second = getSecondChild(children, rule, eh);
        return AttributeConditionTreeBuilder.getRandomAttributeConditionOperator(occurrences, first, second);
    }

    private static AttributeCondition getSecondChild(AttributeCondition[] children, Rule rule, EventHandler eh) {
        return children.length >= 2 ? children[1] : AttributeConditionTreeBuilder.buildAttributeConditionTree(rule.getEventConditionTreeRoot(), 1, eh);
    }
}
