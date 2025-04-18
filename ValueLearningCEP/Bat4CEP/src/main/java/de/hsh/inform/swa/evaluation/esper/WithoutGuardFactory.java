package de.hsh.inform.swa.evaluation.esper;

import java.util.List;

import com.espertech.esper.epl.expression.core.ExprConstantNodeImpl;
import com.espertech.esper.epl.expression.core.ExprNode;
import com.espertech.esper.pattern.EvalStateNodeNumber;
import com.espertech.esper.pattern.MatchedEventConvertor;
import com.espertech.esper.pattern.MatchedEventMap;
import com.espertech.esper.pattern.PatternAgentInstanceContext;
import com.espertech.esper.pattern.guard.Guard;
import com.espertech.esper.pattern.guard.GuardFactorySupport;
import com.espertech.esper.pattern.guard.GuardParameterException;
import com.espertech.esper.pattern.guard.Quitable;
/**
 * Factory class that is called by the engine and instantiates the without pattern.
 * @author Software Architecture Research Group
 *
 */
public class WithoutGuardFactory extends GuardFactorySupport{

	private Object negatedType = null;
	
	@Override
	public void setGuardParameters(List<ExprNode> guardParameters, MatchedEventConvertor convertor)
			throws GuardParameterException {
		negatedType =((ExprConstantNodeImpl)guardParameters.get(0)).getConstantValue(null);
	}

	@Override
	public Guard makeGuard(PatternAgentInstanceContext context, MatchedEventMap beginState, Quitable quitable,
			EvalStateNodeNumber stateNodeId, Object guardState) {
		return new WithoutGuard(negatedType);
	}

}
