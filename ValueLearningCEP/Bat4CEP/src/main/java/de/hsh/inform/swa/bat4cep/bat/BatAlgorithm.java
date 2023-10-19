package de.hsh.inform.swa.bat4cep.bat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import de.hsh.inform.swa.bat4cep.bat.update.PointUpdate;
import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.evaluation.RuleEvaluator;
import de.hsh.inform.swa.evaluation.RuleWithFitness;
import de.hsh.inform.swa.util.EventHandler;
import de.hsh.inform.swa.util.FitnessHelper;
import de.hsh.inform.swa.util.SimpleLogger;
import de.hsh.inform.swa.util.builder.WindowBuilder;
/**
 * Core Class.
 * This is where the bat algorithm takes place. The description of the algorithm can be found in the paper by R. Bruns and J. Dunkel.
 * @author Software Architecture Research Group
 *
 */
public class BatAlgorithm {
    private final Comparator<Bat> BatCMP = (b1, b2) -> b2.compareTo(b1);
    
    //Hyper parameters
    private final double F1_NEARLY_PERFECT_MAXIMUM = 0.995;
    private final double PERCENTAGE_BEST_BAT = 0.05;
    
    private final Bat[] SWARM;
    private final int TIMESTEPS;
    private final double MAX_FREQUENCY, MIN_FREQUENCY, LOUDNESS, PULSERATE, GAMMA, ALPHA;

    private final EventHandler eh;
    private final PointUpdate pu;
    private final RuleEvaluator evaluator;

	private final SimpleLogger log;

    public BatAlgorithm(Bat[] swarm, EventHandler eh, WindowBuilder wb, RuleEvaluator evaluator, PointUpdate pu, int swarmSize, int timesteps, double loudness,
            double alpha, double maxFrequency, double minFrequency, double pulserate, double gamma, SimpleLogger log) {
        this.eh = eh;
        this.SWARM = swarm;
        this.LOUDNESS = loudness;
        this.ALPHA = alpha;
        this.PULSERATE = pulserate;
        this.GAMMA = gamma;
        this.MAX_FREQUENCY = maxFrequency;
        this.MIN_FREQUENCY = minFrequency;
        this.TIMESTEPS = timesteps;
        this.pu = pu;
        this.evaluator = evaluator;
        this.log = log;
    }

    private void measureFitness() {
        RuleWithFitness[] rules = Arrays.stream(SWARM).map(b -> (Rule) b.getSolution()).toArray(size -> new RuleWithFitness[size]);
        FitnessHelper.measureFitness(evaluator, rules);
    }

    public List<Bat> execute() throws InterruptedException, ExecutionException {
        List<Bat> ELITE = new ArrayList<>(); //list of the best bats of each sub-swarm
        ArrayList<Integer> indizes = new ArrayList<>();
        Arrays.sort(SWARM, BatCMP);
        
        //initialization. Determine which bats form a sub-swarm.
        //Bats of the same sub-swarm have the same maximum.
        if(ELITE.isEmpty()) {
        	ArrayList<Integer> shuffled = new ArrayList<>();
        	for(int i=0; i<SWARM.length*PERCENTAGE_BEST_BAT+1;i++) {
            	ELITE.add(SWARM[i].copy());
            }
        	int swarmCount=0, subSwarmCount=0;
        	while(swarmCount<SWARM.length) {
        		if(subSwarmCount<ELITE.size()-1) {
        			subSwarmCount++;
        		}else {
        			subSwarmCount=0;
        		}
        		shuffled.add(subSwarmCount);
        		swarmCount++;
        	}
        	Collections.shuffle(shuffled, new Random(5)); 
        	indizes.addAll(shuffled);
        }
        /*
         * start of the Bat4CEP algorithm presented in the Bat4CEP paper.
         * 
         * note: for performance reasons, the algorithm is divided into three parts in this implementation:
         * 1. generate all intermediate results
         * 2. evaluate all intermediate results at once, so that the engine only has to be called once
         * 3. check for better bats.
         * In the original algorithm intermediate values are not collected but evaluated individually.
         */
        double avgLoudness = BatUtils.getAvgLoudness(SWARM);
        double avgFitness = BatUtils.getAvgFitness(SWARM);
        measureFitness();
        
        long sum_rules=0;
        //step 1: generate
        for (int t = 1; t <= TIMESTEPS; t++) {
        	Map<Integer, List<Bat>> resultsRandomFlight = new HashMap<>();
        	Map<Integer,Bat> resultsLocalSearch = new HashMap<>();
        	Map<Integer,Integer> mapBatIdToBestBatId = new HashMap<>();
            for(int i=0; i<SWARM.length;i++) {
                Bat elite_bat = ELITE.get(indizes.get(i));
                mapBatIdToBestBatId.put(i, indizes.get(i));

                SWARM[i].setF(MIN_FREQUENCY + ((MAX_FREQUENCY - MIN_FREQUENCY) * ThreadLocalRandom.current().nextDouble()));
                SWARM[i].setV(SWARM[i].getV() + (SWARM[i].getSolution().getTotalFitness() - elite_bat.getSolution().getTotalFitness()) * SWARM[i].getF());
             
                randomFlight(SWARM[i], avgFitness, i, resultsRandomFlight);
                
                if (ThreadLocalRandom.current().nextDouble() > SWARM[i].getR()) {
                	localSearch(elite_bat, avgLoudness,i, resultsLocalSearch);
                }

            }
            //step 2: evaluate
            sum_rules += evaluate(resultsRandomFlight, resultsLocalSearch);
            //step 3: check
            for(int i=0; i<SWARM.length;i++) {
            	double ARand = ThreadLocalRandom.current().nextDouble(LOUDNESS);
                synchronized (ELITE) {
                	if (ARand < SWARM[i].getA() && SWARM[i].getSolution().getTotalFitness() > ELITE.get(mapBatIdToBestBatId.get(i)).getSolution().getTotalFitness()) {
                		ELITE.set(mapBatIdToBestBatId.get(i), SWARM[i].copy());
                        updatePulsrateAndLoudness(SWARM[i], t);
                    }
                }
            }
            avgLoudness = BatUtils.getAvgLoudness(SWARM);
            avgFitness = BatUtils.getAvgFitness(SWARM);
            
            //look for the best bat
            Bat bestBat = null;
            for(int i=0; i<ELITE.size();i++) {
            	if(bestBat==null || ELITE.get(i).getSolution().getTotalFitness()>bestBat.getSolution().getTotalFitness()) {
            		bestBat = ELITE.get(i);
            	}
            }
            log.println(String.format(Locale.US, "TIMESTEP: %03d - avg fitness of swarm: %.5f - current best solution: %s", t, avgFitness, bestBat.getSolution()));
            // terminate a run if a nearly perfect solution is found
            if (bestBat.getSolution().getTotalFitness() >= F1_NEARLY_PERFECT_MAXIMUM) {
                break;
            }
            
            
        }
        log.println(String.format("sum of all generated and evaluated rules: %s", sum_rules));
        Collections.sort(ELITE, BatCMP);
        Arrays.sort(SWARM, BatCMP);
        return Arrays.asList(SWARM);
    }
    /**
     * Evaluates all passed rules.
     * @param Rules created by the random flight.
     * @param Rules created by the local search.
     * @return Number of evaluated rules
     */
    private long evaluate(Map<Integer, List<Bat>> mapRandomFlight, Map<Integer, Bat> mapLocalSearch) {
    	ArrayList<RuleWithFitness> allRules = new ArrayList<>();
    	allRules.addAll(mapRandomFlight.values().stream().flatMap(List::stream).map(bat -> bat.getSolution()).collect(Collectors.toCollection(ArrayList::new)));
    	allRules.addAll(mapLocalSearch.values().stream().map(bat -> bat.getSolution()).collect(Collectors.toCollection(ArrayList::new)));

		//call the engine
    	FitnessHelper.measureFitness(evaluator, allRules.toArray(new RuleWithFitness[allRules.size()]));
    	int counter = 0;
    	//random flight: check if any bat got to a better position. If so, update the position of this bat.
    	for(int i=0; i<SWARM.length;i++) {
    		if(mapRandomFlight.containsKey(i)) {
    			for(Bat b: mapRandomFlight.get(i)) {
    				if(allRules.get(counter).getTotalFitness()>=SWARM[i].getSolution().getTotalFitness()) {
    					b.setSolution(allRules.get(counter));
    					SWARM[i] = b.copy();
    				}
    				counter++;
    			}
    		}
    	}
    	//local search: check if any bat got to a better position. If so, update the position of this bat.
    	for(int i=0; i<SWARM.length;i++) {
    		if(mapLocalSearch.containsKey(i)) {
    			if(allRules.get(counter).getTotalFitness()>=SWARM[i].getSolution().getTotalFitness()) {
    				mapLocalSearch.get(i).setSolution(allRules.get(counter));
    				SWARM[i] = mapLocalSearch.get(i).copy();
    			}
    			counter++;
    		}
    	}
    	return allRules.size();
	}

    private void randomFlight(Bat bat, double avgFitness, int i, Map<Integer, List<Bat>> map_optimize) {
        if (bat.getSolution().getTotalFitness() > avgFitness) {
        	map_optimize.put(i, new ArrayList<>());
            optimizeSolution(bat, map_optimize.get(i));
        } else {
            updateSolution(bat);
        }
    }

    private void updatePulsrateAndLoudness(Bat bat, int t) {
    	bat.setR(PULSERATE * (1 - Math.exp(-GAMMA * t)));
        bat.setA(ALPHA * bat.getA());
    }

    private void localSearch(Bat selectedBestBat, double _avgLoudness, int index, Map<Integer, Bat> map_localsearch) {
        Bat bestCopy = selectedBestBat.copy();
        double avgLoudness = ThreadLocalRandom.current().nextDouble() * _avgLoudness;
                
        List<Consumer<PointUpdate>> list = new ArrayList<>();
        list.add((pu) -> pu.explicitComplexityUpdate(bestCopy.getSolution()));
        list.add((pu) -> {
        	if (ThreadLocalRandom.current().nextDouble() < 0.7) {
        		pu.windowRadiusUpdate(bestCopy.getSolution(), F1_NEARLY_PERFECT_MAXIMUM, 0);
        	} else {
        		pu.explicitWindowUpdate(bestCopy.getSolution());
        }});
        if(pu.getMaxACTHeight()>0) list.add((pu) -> pu.updateACT(bestCopy.getSolution(), this.eh));
        int randomNumber = ThreadLocalRandom.current().nextInt(list.size());
        
        for(double j=0.0; j < avgLoudness; j+=0.1) {
        	list.get(randomNumber).accept(pu);

        }   
        map_localsearch.put(index, bestCopy);
    }

    private void updateSolution(Bat bat) {	//no evaluation needed
        while (bat.getV() < bat.getF()) {
            if (ThreadLocalRandom.current().nextDouble(LOUDNESS) < bat.getA() || pu.getMaxACTHeight()==0) {
                pu.updateECTOrWindow(bat.getSolution());
            } else {
                pu.updateACT(bat.getSolution(), this.eh);
            }
            bat.setV(bat.getV() + 0.1);
        }
    }
    //
    private void optimizeSolution(Bat bat, List<Bat> list) {
        Bat copy = bat.copy();
        list.add(copy.copy());
        while (bat.getV() < bat.getF()) {
        	//evaluation later on
            pu.explicitWindowUpdate(copy.getSolution());
            list.add(copy.copy());
            pu.updateECT(copy.getSolution());
            list.add(copy.copy());
            pu.explicitComplexityUpdate(copy.getSolution());
            list.add(copy.copy());
            bat.setV(bat.getV() + 0.1);
        }
    }

}