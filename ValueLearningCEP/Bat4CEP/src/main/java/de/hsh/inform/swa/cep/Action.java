package de.hsh.inform.swa.cep;
/**
 * Class representing an activation, i.e. a detected complex event (usually called HIT).
 * @author Software Architecture Research Group
 *
 */
public class Action {
    private final Event complexEvent;

    public Action(Event complexEvent) {
        this.complexEvent = complexEvent;
    }

    @Override
    public String toString() {
        return getComplexEvent().toString();
    }

    public Action copy() {        
        return new Action(getComplexEvent());
    }

	public Event getComplexEvent() {
		return complexEvent;
	}
}
