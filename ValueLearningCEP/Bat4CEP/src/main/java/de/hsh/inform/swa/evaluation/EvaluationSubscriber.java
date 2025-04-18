package de.hsh.inform.swa.evaluation;

import de.hsh.inform.swa.evaluation.esper.EsperSubscriber;
import de.hsh.inform.swa.util.EventHandler;

/**
 * Subscriber class that receives all events from the rule it was registered with 
 * and determines ...
 * ... the number of hits
 * ... the true positives
 * ... the false positives 
 * @author Software Architecture Research Group
 *
 */
public class EvaluationSubscriber extends EsperSubscriber{
    private EventHandler eh;

    public EvaluationSubscriber(EventHandler eh) {
        this.eh = eh;
    }
    public int getTruePositives() {
        int result = 0;
        for (Integer s : super.getFiredPosition()) {
            if (s < eh.getEventDataSize()) {
                if (eh.getEventData().get(s).getType().equals(eh.getComplexEvent().getType())) {
                    result++;
                }
            }
        }
        return result;
    }

    public int getFalsePositives() {
        int result = 0;
        for (Integer s : super.getFiredPosition()) {
            if (s < eh.getEventDataSize()) {
                if (!eh.getEventData().get(s).getType().equals(eh.getComplexEvent().getType())) {
                    result++;
                }
            }
        }
        return result;
    }
    @Override
    public int getOffset() {
    	return 1;
    }
}
