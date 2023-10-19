package de.hsh.inform.swa.util.builder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import com.espertech.esper.util.TriFunction;

import de.hsh.inform.swa.cep.Attribute;
import de.hsh.inform.swa.cep.AttributeCondition;
import de.hsh.inform.swa.cep.AttributeOperator;
import de.hsh.inform.swa.cep.Condition;
import de.hsh.inform.swa.cep.ConstantAttribute;
import de.hsh.inform.swa.cep.EventCondition;
import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.cep.TemplateEvent;
import de.hsh.inform.swa.cep.operators.attributes.aggregation.AggregationAttribute;
import de.hsh.inform.swa.cep.operators.attributes.aggregation.SumAggregateAttribute;
import de.hsh.inform.swa.cep.operators.attributes.arithmetic.ArithmeticOperator;
import de.hsh.inform.swa.util.EventHandler;

/**
 * Class that builds the ACT tree.
 * 
 * A second task of this class is to repair the ACT. This is needed when an ECT operation removes an event 
 * while its attribute is still referenced in the ACT or to ensure that all aggregation functions in the act have the same time window.
 * @author Software Architecture Research Group
 *
 */
public class AttributeConditionTreeBuilder {

    private static final int MAX_CONDITION_TREE_OCCURENCE = 1;
	private static final List<TriFunction<Map<TemplateEvent, Integer>, AttributeCondition, AttributeCondition, AttributeCondition>> attributeConditionBuilderList = new ArrayList<>();

    static {
        attributeConditionBuilderList.add((occ, left, right) -> buildRandomAttributeComparisonOperator(occ));
        attributeConditionBuilderList.add((occ, left, right) -> AttributeConditionTreeOperators.getRandomLogicalOperator(left, right));
    }

    public static AttributeCondition buildAttributeConditionTree(EventCondition ect, int maxHeight, EventHandler eh) {
        Map<TemplateEvent, Integer> occ = ConditionTreeTraverser.getTemplateEventsOccurrencesOfEct(ect, eh);
        return buildRandomAttributeCondition(occ, maxHeight);
    }

    public static AttributeCondition buildRandomAttributeCondition(Map<TemplateEvent, Integer> occurrences, int maxHeight) {
        if (maxHeight == 0) return buildRandomAttributeComparisonOperator(occurrences);
        return getRandomAttributeConditionOperator(occurrences, buildRandomAttributeCondition(occurrences, maxHeight-1), buildRandomAttributeCondition(occurrences, maxHeight-1));        
    }

    public static AttributeOperator buildRandomAttributeComparisonOperator(Map<TemplateEvent, Integer> occurrences) {
        Attribute firstOperand = AttributeConditionTreeOperators.getRandomEventAttributeOfEct(occurrences);
        return buildRandomAttributeComparisonOperator(occurrences, firstOperand);
    }
    public static AttributeCondition getRandomAttributeConditionOperator(Map<TemplateEvent, Integer> occ, AttributeCondition firstOperand, AttributeCondition secondOperand) {
    	int operatorOption = ThreadLocalRandom.current().nextInt(attributeConditionBuilderList.size());
    	return attributeConditionBuilderList.get(operatorOption).apply(occ, firstOperand, secondOperand);
	}

	public static AttributeOperator buildRandomAttributeComparisonOperator(Map<TemplateEvent, Integer> occurrences, Attribute firstOperand) {
        Attribute secondOperand;
        do {
        	secondOperand = buildRandomSecondAttribute(firstOperand, occurrences);
        }while(firstOperand.equals(secondOperand)); // avoiding conditions like C0.c1 = C0.c1
        return AttributeConditionTreeOperators.getRandomComparisonOperator(firstOperand, secondOperand);
    }

    public static Attribute buildRandomSecondAttribute(Attribute firstOperand, Map<TemplateEvent, Integer> occurrences) {
        if (ThreadLocalRandom.current().nextDouble() < 0.5) {
        	Attribute attr;
        	do { 
        		attr = AttributeConditionTreeOperators.getRandomEventAttributeOfEct(occurrences);
        	}while(attr.equals(firstOperand)); // avoiding useless conditions like C0.c1 = C0.c1
            return attr;
        }
        return AttributeConditionTreeOperators.getRandomConstantAttribute(firstOperand);
    }

    public static void repairAct(Rule rule, EventHandler eh) {
        // Tree run - to remove duplicates bigger than MAX_CONDITION_TREE_OCCURENCE
        ConditionTreeTraverser.removeDuplicates(rule, rule.getEventConditionTreeRoot(), MAX_CONDITION_TREE_OCCURENCE);
        ConditionTreeTraverser.simplfyACTor(rule, rule.getAttributeConditionTreeRoot(), MAX_CONDITION_TREE_OCCURENCE);
        ConditionTreeTraverser.simplfyACT(rule, rule.getAttributeConditionTreeRoot(), MAX_CONDITION_TREE_OCCURENCE);

        AttributeCondition act = rule.getAttributeConditionTreeRoot();
        if (act != null) {
            EventCondition ect = rule.getEventConditionTreeRoot();
            Set<String> usedAliasesInEct = ConditionTreeTraverser.getAllEventAliasesUnderEventCondition(ect);
            Set<String> usedAliasesInAct = ConditionTreeTraverser.getAllUsedAliases(act);

            usedAliasesInAct.removeAll(usedAliasesInEct);
            if (!usedAliasesInAct.isEmpty()) {
                Set<AttributeOperator> brokenComparisonOperators = ConditionTreeTraverser
                        .getAllAttributeComparisonOperatorsUsingBrokenAlias(usedAliasesInAct, act);
                for (AttributeOperator brokenOperator : brokenComparisonOperators) {
                    repairOperator(brokenOperator, usedAliasesInAct, ect, eh);
                }
            }
        }
    }

    private static void repairOperator(AttributeOperator brokenOperator, Set<String> usedAliasesInAct, EventCondition ect, EventHandler eh) {
        Attribute[] operands = brokenOperator.getOperands();
        Attribute firstOperand = operands[0];
        Attribute secondOperand = operands[1];
        
        boolean arithmeticFirstOperand = firstOperand instanceof ArithmeticOperator &&
        		Arrays.stream(((ArithmeticOperator)firstOperand).getOperands()).map(x -> usedAliasesInAct.contains(x.getAlias())).anyMatch(Boolean.TRUE::equals);
        
        boolean arithmeticSecondOperand = secondOperand instanceof ArithmeticOperator &&
        		Arrays.stream(((ArithmeticOperator)secondOperand).getOperands()).map(x -> usedAliasesInAct.contains(x.getAlias())).anyMatch(Boolean.TRUE::equals);
        
        Map<TemplateEvent, Integer> occurrences = ConditionTreeTraverser.getTemplateEventsOccurrencesOfEct(ect, eh);
        if (arithmeticFirstOperand || usedAliasesInAct.contains(firstOperand.getAlias())) {
            firstOperand = AttributeConditionTreeOperators.getRandomEventAttributeOfEct(occurrences);
            brokenOperator.setOperand(firstOperand, 0);
        } 
        if (operands[1] instanceof ConstantAttribute) {
            double valueOfConstant = ((ConstantAttribute) operands[1]).getValue();
            double min = firstOperand.getMin();
            double max = firstOperand.getMax();
            if (valueOfConstant < min || valueOfConstant > max) {
                brokenOperator.setOperand(AttributeConditionTreeOperators.getRandomConstantAttribute(firstOperand), 1);
            }
        } else if (arithmeticSecondOperand || usedAliasesInAct.contains(secondOperand.getAlias())) {
            brokenOperator.setOperand(buildRandomSecondAttribute(firstOperand, occurrences), 1);
        }
    }
    // due to implementation-specific properties of Esper aggregation functions, 
    // this method ensures that all time windows within a rule are identical.
    public static void repairAggregationWindowsInAct(Rule rule, EventHandler eh) {
    	if (eh != null) repairAct(rule, eh);
        Condition currentPosInAct = rule.getAttributeConditionTreeRoot();
        int i=1;
        while(currentPosInAct != null) {
        	if(currentPosInAct instanceof AttributeOperator) {
        		AttributeOperator currentCompareOperator = (AttributeOperator) currentPosInAct;
          		Attribute[] attr = currentCompareOperator.getOperands();
          		if(attr[0] instanceof AggregationAttribute) {
        				AggregationAttribute curAggrAttribute = (AggregationAttribute)attr[0];
        				curAggrAttribute.setWindow(rule.getWindow().copy());
        				/*
        				 * the sum operator requires special treatment. Be for example:
        				 * current rule x) ACT: [a0=A -> b0=B], ECT: [SUM(A)>50], WINDOW: [10 sec]
        				 * wanted rule x*) ACT: [a0=A -> b0=B], ECT: [SUM(A)>500], WINDOW: [100 sec]
        				 * With the current algorithm, it is not possible to arrive at position x*
        				 * since the operations of Bat4CEP do not consider the relationship 
        				 * between the sum constant and the window. For the rule x, sum(A)>50 and window=[10 sec]
        				 * are in the optimal ratio. If we change the window, the sum operator would include more events,
        				 * while the constant 50 remains the same. The overall result would be worse and the changes would be discarded.
        				 */
        				if(curAggrAttribute instanceof SumAggregateAttribute && attr[1] instanceof ConstantAttribute && curAggrAttribute.getWindow().getValue()>0) {
        					if(ThreadLocalRandom.current().nextDouble()>0.5) {
            					long oldWindow = curAggrAttribute.getWindow().getValue();
                				long newWindow = rule.getWindow().getValue();
            					double oldValue = ((ConstantAttribute)attr[1]).getValue();
        						double relation = (double) newWindow / oldWindow;	
        						double newValue=relation * oldValue;
        						currentCompareOperator.setOperand(new ConstantAttribute(newValue), 1);
        					} 
        				}
        			}
          		if(attr[1] instanceof AggregationAttribute) {
          			AggregationAttribute curAggrAttribute = (AggregationAttribute)attr[1];
    				curAggrAttribute.setWindow(rule.getWindow().copy());
          		}
            	}
              currentPosInAct = ConditionTreeTraverser.getConditionWithPreOrderIndex(rule.getAttributeConditionTreeRoot(), i++);
          }
      }
}
