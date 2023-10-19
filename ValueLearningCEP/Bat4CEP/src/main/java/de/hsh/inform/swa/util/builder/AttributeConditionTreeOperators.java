package de.hsh.inform.swa.util.builder;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.BiFunction;
import java.util.function.Function;

import de.hsh.inform.swa.cep.Attribute;
import de.hsh.inform.swa.cep.AttributeCondition;
import de.hsh.inform.swa.cep.AttributeOperator;
import de.hsh.inform.swa.cep.ConstantAttribute;
import de.hsh.inform.swa.cep.EventAttribute;
import de.hsh.inform.swa.cep.TemplateEvent;
import de.hsh.inform.swa.cep.operators.attributes.aggregation.AggregationAttribute;
import de.hsh.inform.swa.cep.operators.attributes.aggregation.AvgAggregateAttribute;
import de.hsh.inform.swa.cep.operators.attributes.aggregation.MaxAggregateAttribute;
import de.hsh.inform.swa.cep.operators.attributes.aggregation.MinAggregateAttribute;
import de.hsh.inform.swa.cep.operators.attributes.aggregation.SumAggregateAttribute;
import de.hsh.inform.swa.cep.operators.attributes.arithmetic.AdditionOperator;
import de.hsh.inform.swa.cep.operators.attributes.arithmetic.ArithmeticOperator;
import de.hsh.inform.swa.cep.operators.attributes.arithmetic.SubtractionOperator;
import de.hsh.inform.swa.cep.operators.attributes.comparison.EqualToAttributeComparisonOperator;
import de.hsh.inform.swa.cep.operators.attributes.comparison.GreaterThanAttributeComparisonOperator;
import de.hsh.inform.swa.cep.operators.attributes.comparison.LessThanAttributeComparisonOperator;
import de.hsh.inform.swa.cep.operators.attributes.logic.AndAttributeOperator;
import de.hsh.inform.swa.cep.operators.attributes.logic.NotAttributeOperator;
import de.hsh.inform.swa.cep.operators.attributes.logic.OrAttributeOperator;
/**
 * Class that defines all legal operands in the ACT.
 * @author Software Architecture Research Group
 *
 */
public class AttributeConditionTreeOperators {
	
	private static final double CONSTANT_ATTRIBUTE_STEPS = 10.0; // has to be positive (>0).
	private static final double AGGREGATION_OCCURENCE_RATE = 0.2;
	private static final double ARITHMETIC_OCCURENCE_RATE = 0.3;
	
	private static final List<BiFunction<AttributeCondition, AttributeCondition, AttributeCondition>> logicalOperatorList = new ArrayList<>();
	private static final List<Function<EventAttribute, AggregationAttribute>> aggregationOperatorList = new ArrayList<>();
	private static final List<BiFunction<Attribute, Attribute, AttributeOperator>> comparisonOperatorlist = new ArrayList<>();
	private static final List<BiFunction<Attribute, Attribute, ArithmeticOperator>> arithmeticOperatorList = new ArrayList<>();

	static {
        logicalOperatorList.add((left, right) -> new AndAttributeOperator(left, right));
        logicalOperatorList.add((left, right) -> new NotAttributeOperator(left));
        logicalOperatorList.add((left, right) -> new OrAttributeOperator(left, right));
        
        comparisonOperatorlist.add((left, right) -> new EqualToAttributeComparisonOperator(left, right));
        comparisonOperatorlist.add((left, right) -> new LessThanAttributeComparisonOperator(left, right));
        comparisonOperatorlist.add((left, right) -> new GreaterThanAttributeComparisonOperator(left, right));
        
        aggregationOperatorList.add((attr) -> new AvgAggregateAttribute(attr));
        aggregationOperatorList.add((attr) -> new SumAggregateAttribute(attr));
        aggregationOperatorList.add((attr) -> new MaxAggregateAttribute(attr));
        aggregationOperatorList.add((attr) -> new MinAggregateAttribute(attr));
        
        arithmeticOperatorList.add((left, right) -> new AdditionOperator(left, right));
        arithmeticOperatorList.add((left, right) -> new SubtractionOperator(left, right));
    }
	
	private static EventAttribute getRandomAggregateAttribute(EventAttribute firstOperand) {
    	int operatorOption = ThreadLocalRandom.current().nextInt(aggregationOperatorList.size());
    	return aggregationOperatorList.get(operatorOption).apply(firstOperand);
	}
    
    public static AttributeCondition getRandomLogicalOperator(AttributeCondition firstOperand, AttributeCondition secondOperand) {
    	int operatorOption = ThreadLocalRandom.current().nextInt(logicalOperatorList.size());
    	return logicalOperatorList.get(operatorOption).apply(firstOperand, secondOperand);
	}
    
    public static AttributeOperator getRandomComparisonOperator(Attribute firstOperand, Attribute secondOperand) {
    	int operatorOption = ThreadLocalRandom.current().nextInt(comparisonOperatorlist.size());
    	return comparisonOperatorlist.get(operatorOption).apply(firstOperand, secondOperand);
	}
    
    public static ArithmeticOperator getRandomArithmeticOperator(Attribute firstOperand, Attribute secondOperand) {
    	int operatorOption = ThreadLocalRandom.current().nextInt(arithmeticOperatorList.size());
    	return arithmeticOperatorList.get(operatorOption).apply(firstOperand, secondOperand);
	}
    
    public static Attribute getRandomEventAttributeOfEct(Map<TemplateEvent, Integer> occurrences) {
    	TemplateEvent rndTe = getRandomTemplateEvent(occurrences);
        EventAttribute firstOperand =  getRandomAttributeOfTemplateEvent(rndTe, occurrences.get(rndTe));
        if(ThreadLocalRandom.current().nextDouble()<AGGREGATION_OCCURENCE_RATE) {
        	return getRandomAggregateAttribute(firstOperand);
    	}else if(ThreadLocalRandom.current().nextDouble()<ARITHMETIC_OCCURENCE_RATE) {
    		TemplateEvent rndTe2 = getRandomTemplateEvent(occurrences);
    		Attribute secondOperand = getRandomAttributeOfTemplateEvent(rndTe2, occurrences.get(rndTe2));
    		return getRandomArithmeticOperator(firstOperand, secondOperand);
    	}
        return firstOperand;
    }

    private static TemplateEvent getRandomTemplateEvent(Map<TemplateEvent, Integer> occurrences) {
        int idx = ThreadLocalRandom.current().nextInt(occurrences.keySet().size());
        Iterator<TemplateEvent> itr = occurrences.keySet().iterator();
        for (int i = 0; i < idx; i++) {
            itr.next();
        }
        return itr.next();
    }
    
    private static EventAttribute getRandomAttributeOfTemplateEvent(TemplateEvent te, int occurrence) {
        String alias = te.getType() + ThreadLocalRandom.current().nextInt(occurrence);
        int idx = ThreadLocalRandom.current().nextInt(te.getAttributes().size());
        Iterator<String> attrItr = te.getAttributes().iterator();
        for (int i = 0; i < idx; i++) {
            attrItr.next();
        }
        String attributeName = attrItr.next();
        return new EventAttribute(alias, attributeName, te);
    }
    
    public static ConstantAttribute getRandomConstantAttribute(Attribute firstOperand) {
        double min = firstOperand.getMin();
        double max = firstOperand.getMax();
        if(min>=max) max = min+1;
        double value = Math.floor(ThreadLocalRandom.current().nextDouble(min, max)/CONSTANT_ATTRIBUTE_STEPS)*CONSTANT_ATTRIBUTE_STEPS;        
        return new ConstantAttribute(value);
    }
    
}
