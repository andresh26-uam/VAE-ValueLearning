package de.hsh.inform.swa.evaluation.esper;

import java.util.IntSummaryStatistics;
import java.util.List;
import com.espertech.esper.event.map.MapEventBean;
import com.espertech.esper.pattern.MatchedEventMap;
import com.espertech.esper.pattern.guard.EventGuardVisitor;
import com.espertech.esper.pattern.guard.Guard;
import de.hsh.inform.swa.cep.Event;

/**
 * Pattern guard that emulates the behavior of the "AND NOT" operator.
 * 
 * Emulation:
 * Let's take the expression [(A -> C) where cep:without(B)] as an example. When the inner clause matches, 
 * the pattern guard looks at all events between the matched A and C events and checks if there is a B event in between.
 * @author Software Architecture Research Group
 *
 */
public class WithoutGuard implements Guard{

	private String notType;
	private static List<Event> events;

	public WithoutGuard(Object negatedType) {
		this.notType = (String) negatedType;
	}

	@Override
	public boolean inspect(MatchedEventMap matchEvent) {
		if(matchEvent.getMatchingEvents().length==1) return true;
		IntSummaryStatistics  stream = matchEvent.getMatchingEventsAsMap().values().stream().mapToInt(bean -> (Integer)((MapEventBean) bean).get("_lineNumber")).summaryStatistics();
		
		int max = stream.getMax();
		int min = stream.getMin();
		
		for(int i=min+1; i<max;i++) {
			if(WithoutGuard.events.get(i).getType().equals(notType)) {
				return false;
			}
		}
		return true;
	}
	
	public static void initEvents(List<Event> events) {
		WithoutGuard.events = events;
	}

	@Override
	public void accept(EventGuardVisitor visitor) {}
	
	@Override
	public void startGuard() {}

	@Override
	public void stopGuard() {}

}
