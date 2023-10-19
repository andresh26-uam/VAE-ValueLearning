package de.hsh.inform.swa.util.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.espertech.esper.collection.Pair;

import de.hsh.inform.swa.cep.Event;
/**
 * This class reads the traffic data and converts them into events.
 * @author Software Architecture Research Group
 *
 */
public class DataCreatorTraffic {
    static Pair<List<Event>, List<Event>> readTrafficData(DatasetEnum dataset) throws ParseException {
    	List<Event> trafficEvents = new ArrayList<>();
    	List<Event> trafficEventsHoldout = new ArrayList<>();
    	SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    	String csvFile;
    	if(dataset == DatasetEnum.TRAFFIC_FIRST_HALF) {
    		csvFile = "data/madrid_traffic_1.csv";
    	}else { // dataset == DatasetEnum.TRAFFIC_SECOND_HALF
    		csvFile = "data/madrid_traffic_2.csv";
    	}
         String line = "";
         String cvsSplitBy = ";";

         int cnt=0;
         try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
        	 br.readLine();
             while ((line = br.readLine()) != null) {
            	String[] row = line.split(cvsSplitBy);
                Date time = formatter.parse(row[0]);
     	 		Map<String, Object> attributes = new HashMap<>();
     	 		attributes.put("INTENSITY", Integer.parseInt(row[1]));
     	 		attributes.put("OCUPATION", Integer.parseInt(row[2]));
     	 		attributes.put("VMED", Integer.parseInt(row[3]));
     	 		Event ev = new Event("A", time, attributes);	// the data stream consists of only one event type
     	 		if (cnt%2==0) { // add every second row to the training data set. The rest is assigned to the test data set
     	 			trafficEvents.add(ev);
     	 		}else {
     	 			trafficEventsHoldout.add(ev);
     	 		}
     	 		cnt++;
             }
             br.close();
         } catch (IOException e) {
             e.printStackTrace();
         }
         
         return new Pair<List<Event>, List<Event>>(trafficEvents, trafficEventsHoldout);
    }
    static Pair<List<Event>,List<Event>> getTrafficEventsWithHits(DataCreatorConfig config){
    	Pair<List<Event>, List<Event>> allEvents;
		try {
			allEvents = readTrafficData(config.getData());
		} catch (ParseException e) {
			throw new RuntimeException(e);
		}
    	List<Event> allEventsTraining = DataCreator.addComplexEventsToStream(allEvents.getFirst(), config.getRule());
        List<Event> allEventsHoldout = DataCreator.addComplexEventsToStream(allEvents.getSecond(), config.getRule());
    	return new Pair<List<Event>, List<Event>>(allEventsTraining, allEventsHoldout);
    }
}
