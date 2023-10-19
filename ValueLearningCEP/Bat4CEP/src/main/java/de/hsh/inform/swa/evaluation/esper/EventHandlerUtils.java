package de.hsh.inform.swa.evaluation.esper;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import com.espertech.esper.client.Configuration;

import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.util.EventHandler;
/**
 * Configuration class. Adds event types and pattern guards to the data stream.
 * @author Software Architecture Research Group
 */
public class EventHandlerUtils {

	public static Configuration toEsperConfiguration(EventHandler eventHandler) {
		Configuration configuration = new Configuration();

		Map<String, HashMap<String, Object>> data = new HashMap<>();

		for (Event e : eventHandler.getWithoutComplexEvent()) {
			HashMap<String, Object> typeMap = data.get(e.getType());
			if (typeMap == null) {
				typeMap = new HashMap<>();
				data.put(e.getType(), typeMap);
			}
			for (Entry<String, Object> attr : e.getAttributes().entrySet()) {
				if (!typeMap.containsKey(attr.getKey())) {
					typeMap.put(attr.getKey(), attr.getValue().getClass());
				}

			}
		}

		for (Entry<String, HashMap<String, Object>> entry1 : data.entrySet()) {
			configuration.addEventType(entry1.getKey(), entry1.getValue());
		}
		
		configuration.addEventType(eventHandler.getComplexEvent().getType(), new HashMap<String,Object>());
		
		//register self-written guards
		configuration.addPlugInPatternGuard("cep", "without", WithoutGuardFactory.class.getName());
		
		return configuration;
	}

}
