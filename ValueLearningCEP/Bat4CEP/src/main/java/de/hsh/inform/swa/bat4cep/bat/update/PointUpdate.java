package de.hsh.inform.swa.bat4cep.bat.update;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import de.hsh.inform.swa.cep.AttributeCondition;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.cep.EventCondition;
import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.cep.TemplateEvent;
import de.hsh.inform.swa.cep.operators.attributes.logic.AndAttributeOperator;
import de.hsh.inform.swa.evaluation.RuleWithFitness;
import de.hsh.inform.swa.util.EventHandler;
import de.hsh.inform.swa.util.builder.AttributeConditionTreeBuilder;
import de.hsh.inform.swa.util.builder.ConditionTreeTraverser;
import de.hsh.inform.swa.util.builder.WindowBuilder;
/**
 * Class that coordinates all changes to a given rule. Points for changes are the ACT, ECT and window.
 * Used by BatAlgorithm.java.
 * @author Software Architecture Research Group
 *
 */
public class PointUpdate {

    private final WindowBuilder wb;
    private final EventHandler eh;
    private final EventConditionTreePointUpdate ectUpdate;

    private final int maxECTHeight;
    private final int maxACTHeight;

    public PointUpdate(WindowBuilder wb, List<Event> eventTypes, EventHandler eh, int maxETCHeight, int maxATCHeight) {
        this.wb = wb;
        this.eh = eh;
        ectUpdate = new EventConditionTreePointUpdate(eventTypes);
        this.maxACTHeight = maxATCHeight;
        this.maxECTHeight = maxETCHeight;
    }

    public int getMaxECTHeight(){
    	return this.maxECTHeight;
    }
    public int getMaxACTHeight(){
    	return this.maxACTHeight;
    }

    public void updateECT(Rule r) {
        EventCondition ectRoot = r.getEventConditionTreeRoot();
        int numberOfEctNodes = ectRoot.getNumberOfNodes();
        int updatePoint = ThreadLocalRandom.current().nextInt(numberOfEctNodes);
        ectUpdate.update(r, updatePoint, eh);
    }

    public void explicitComplexityUpdate(Rule r) {
        EventCondition ectRoot = r.getEventConditionTreeRoot();
        int numberOfEctNodes = ectRoot.getNumberOfNodes();
        int updatePoint = ThreadLocalRandom.current().nextInt(numberOfEctNodes);
        if (ThreadLocalRandom.current().nextDouble() < 0.7) {	
            ectUpdate.update(r, updatePoint, eh);
        } else {
            ectUpdate.reduceComplexity(r, updatePoint, eh);
        }
    }

    public void updateECTOrWindow(Rule rule) {
        EventCondition ectRoot = rule.getEventConditionTreeRoot();
        int numberOfEctNodes = ectRoot.getNumberOfNodes() + 1;
        int updatePoint = ThreadLocalRandom.current().nextInt(numberOfEctNodes);
        if (updatePoint == 0) {
            WindowUpdate.update(rule, wb);
        }else if ((--updatePoint)<numberOfEctNodes){
        	ectUpdate.update(rule, updatePoint, eh);
        }
    }
    public void explicitWindowUpdate(Rule r) {
        if (ThreadLocalRandom.current().nextDouble() < 0.7) {
            WindowUpdate.update(r, wb);
        } else {
            WindowUpdate.explicitReduceWindowSize(r);
        }
    }

    public void windowRadiusUpdate(RuleWithFitness r, double functionSpecificMaxFitness, double functionSpecificMinFitness) {
        double radius = ((functionSpecificMaxFitness - r.getTotalFitness()) / (functionSpecificMaxFitness-functionSpecificMinFitness)) * 0.35f;
        WindowUpdate.localRandomWindow(r, wb, radius);
    }

    public void updateACT(Rule r, EventHandler eh) {
    	if(maxACTHeight == 0) return;
        AttributeCondition actRoot = r.getAttributeConditionTreeRoot();
        if (ThreadLocalRandom.current().nextDouble() > 0.7 && r.getAttributeConditionTreeRoot().getHeight() < maxACTHeight) {
            Map<TemplateEvent, Integer> occurrences = ConditionTreeTraverser.getTemplateEventsOccurrencesOfEct(r.getEventConditionTreeRoot(), eh);
            r.setAttributeConditionTreeRoot(
                    new AndAttributeOperator(actRoot, AttributeConditionTreeBuilder.buildRandomAttributeComparisonOperator(occurrences)));
        } else {
            int numberOfActNodes = actRoot.getNumberOfNodes() + 1;
            int updatePoint = ThreadLocalRandom.current().nextInt(numberOfActNodes);
            AttributeConditionTreePointUpdate.update(r, eh, updatePoint);
        }
    }
}
