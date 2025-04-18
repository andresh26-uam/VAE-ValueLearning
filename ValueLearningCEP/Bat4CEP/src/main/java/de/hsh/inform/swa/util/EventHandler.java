package de.hsh.inform.swa.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.cep.TemplateEvent;
/**
 * Entity class that contains all the important information about the basic data stream.
 * @author Software Architecture Research Group
 *
 */
public final class EventHandler {
    private final List<Event> eventData = new ArrayList<>();
    
    private final AtomicReference<Object> indicesOfComplexEvent = new AtomicReference<Object>();
    private final AtomicReference<Object> templateEvents = new AtomicReference<Object>();
    private final AtomicReference<Object> withoutComplexEvent = new AtomicReference<Object>();
    private final AtomicReference<Object> eventTypes = new AtomicReference<Object>();
    private final AtomicReference<Object> complexEventCount = new AtomicReference<Object>();

    private Event complexEvent;

    public EventHandler(List<Event> eventsWithComplexEvent, Event complexEvent) {
        this.eventData.addAll(eventsWithComplexEvent);
        this.complexEvent = complexEvent;
    }

	private Map<String, TemplateEvent> getTemplateEventsInternal() {
        Map<String, TemplateEvent> templateEvents = new HashMap<>();
        for (Event e : getEventData()) {
            if (!e.getType().equals(getComplexEvent().getType())) {
                TemplateEvent te = templateEvents.get(e.getType());
                if (te == null) {
                    te = new TemplateEvent(e);
                    templateEvents.put(e.getType(), te);
                } else {
                    te.updateExtremes(e);
                }
            }
        }
        return templateEvents;

    }

    private List<Event> getWithoutComplexEventInternal() {
        return eventData.stream().filter(e -> !e.getType().equals(getComplexEvent().getType())).collect(Collectors.toList());
    }

    private long getComplexEventCountInternal() {
        return getEventData().stream().filter(e -> e.getType().equals(getComplexEvent().getType())).count();
    }

    private List<Event> getEventTypesInternal() {
        return eventData.stream().map(e -> e.getType()).distinct().filter(type -> !type.equals(getComplexEvent().getType())).map(type -> new Event(type)).collect(Collectors.toList());
    }

    private int[] getIndicesOfComplexEventInternal() {
        List<Integer> indices = new ArrayList<>();
        int index = 0;
        for (Event e : eventData) {
            if (!e.getType().equals(complexEvent.getType())) {
                indices.add(index);
            }
            index++;
        }

        return indices.stream().mapToInt(i -> i).toArray();
    }

    public List<Event> getEventData() {
        return eventData;
    }

    public int getEventDataSize() {
        return eventData.size();
    }

    public Event getComplexEvent() {
        return complexEvent;
    }

    public TemplateEvent getTemplateOfEvent(Event event) {        
        return getTemplateEvents().get(event.getType());
    }
    
	public int[] getIndicesOfComplexEvent() { //lazy delegation method
		java.lang.Object value = this.indicesOfComplexEvent.get();
	    if (value == null) {
	      synchronized(this.indicesOfComplexEvent) {
	        value = this.indicesOfComplexEvent.get();
	        if (value == null) {
	          final int[] actualValue = getIndicesOfComplexEventInternal();
	          value = actualValue == null ? this.indicesOfComplexEvent : actualValue;
	          this.indicesOfComplexEvent.set(value);
	        }
	      }
	    }
	    return (int[])(value == this.indicesOfComplexEvent ? null : value);
	}
	
	@SuppressWarnings("unchecked")
	public Map<String, TemplateEvent> getTemplateEvents() { //lazy delegation method
		java.lang.Object value = this.templateEvents.get();
	    if (value == null) {
	      synchronized(this.templateEvents) {
	        value = this.templateEvents.get();
	        if (value == null) {
	          final Map<String, TemplateEvent> actualValue = getTemplateEventsInternal();
	          value = actualValue == null ? this.templateEvents : actualValue;
	          this.templateEvents.set(value); 
	        }
	      }
	    }  
		return (Map<String, TemplateEvent>)(value == this.templateEvents ? null : value);
	}
	@SuppressWarnings("unchecked")
	public List<Event> getWithoutComplexEvent() { //lazy delegation method
		java.lang.Object value = this.withoutComplexEvent.get();
	    if (value == null) {
	      synchronized(this.withoutComplexEvent) {
	        value = this.withoutComplexEvent.get();
	        if (value == null) {
	          final List<Event> actualValue = getWithoutComplexEventInternal();
	          value = actualValue == null ? this.withoutComplexEvent : actualValue;
	          this.withoutComplexEvent.set(value);
	        }
	      }
	    }
	    return (List<Event>)(value == this.withoutComplexEvent ? null : value);
	}
	@SuppressWarnings("unchecked")
	public List<Event> getEventTypes() { //lazy delegation method
		java.lang.Object value = this.eventTypes.get();
	    if (value == null) {
	      synchronized(this.eventTypes) {
	        value = this.eventTypes.get();
	        if (value == null) {
	          final List<Event> actualValue = getEventTypesInternal();
	          value = actualValue == null ? this.eventTypes : actualValue;
	          this.eventTypes.set(value);
	        }
	      }
	    }
	    return (List<Event>)(value == this.eventTypes ? null : value);
	}
	public Long getComplexEventCount() { //lazy delegation method
		java.lang.Object value = this.complexEventCount.get();
	    if (value == null) {
	      synchronized(this.complexEventCount) {
	        value = this.complexEventCount.get();
	        if (value == null) {
	          final Long actualValue = getComplexEventCountInternal();
	          value = actualValue == null ? this.complexEventCount : actualValue;
	          this.complexEventCount.set(value);
	        }
	      }
	    }
	    return (long)(value == this.complexEventCount ? null : value);
	}
}
