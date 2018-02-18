//username : mzr3
package opt.test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.Distribution;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;



public class TravelingSalesmanProblem {

	private static String outputDir = "./OptimizationResults";
	private static final int N = 50;
	/**
	 * The test main
	 * @param args ignored
	 */
	private static Random random = new Random();
	// create the random points
	private static double[][] points = new double[N][2];
	static {
		for (int i = 0; i < points.length; i++) {

			points[i][0] = random.nextDouble();
			points[i][1] = random.nextDouble();   
		}
	}
	private static int[] ranges = new int[N];
    static {
    	Arrays.fill(ranges, N);
    }

	/** The evaluation function */
	private static TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);

	private static Distribution odd = new DiscretePermutationDistribution(N);

	private static enum Algorithm {
		RHC, SA, GA, MIMIC
	};

	/**
	 * Traveling Salesman Problem optimizations using different algorithms
	 */
	private static class tspOptimization implements Runnable {

		private Algorithm algorithm;

		private HashMap<String, Double> params;

		public tspOptimization(Algorithm algorithm, HashMap<String, Double> params) {
			this.algorithm = algorithm;
			this.params = params;
		}

		private void tspOptimizationRHC() {
			String results = "";

			NeighborFunction nf = new SwapNeighbor();
			HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

			RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);

			double start = System.nanoTime();
			for (int i = 1; i <= params.get("iterations").intValue(); i++) {
				if (i%10 == 0){
				rhc.train();
				results += i + ",";
				results += ef.value(rhc.getOptimal()) + ",";
				results += (System.nanoTime() - start) / Math.pow(10, 9) + "\n";
				}
			}

			Utils.writeOutputToFile(outputDir, "tspRHC.csv", results);
			System.out.println("  RHC terminated run " + params.get("run").intValue());
		}

		private void tspOptimizationSA() {
			String results = "";

			NeighborFunction nf = new SwapNeighbor();
			HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

			SimulatedAnnealing sa = new SimulatedAnnealing(params.get("SA_initial_temperature"),
					params.get("SA_cooling_factor"), hcp);

			double start = System.nanoTime();
			for (int i = 1; i <= params.get("iterations").intValue(); i++) {
				if (i%10 == 0){
				sa.train();
				results += i + ",";
				results += ef.value(sa.getOptimal()) + ",";
				results += (System.nanoTime() - start) / Math.pow(10, 9) + ",";
				results += params.get("SA_initial_temperature") + ",";
				results += params.get("SA_cooling_factor") + "\n";
				}
			}

			Utils.writeOutputToFile(outputDir, "TSPSA_IT" + params.get("SA_initial_temperature") + "_CF"
					+ params.get("SA_cooling_factor") + ".csv", results);
			System.out.println("  SA terminated run " + params.get("run").intValue());
		}

		private void tspOptimizationGA() {
			String results = "";


			MutationFunction mf = new SwapMutation();
			CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
			GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

			StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(params.get("GA_population").intValue(),
					params.get("GA_mate_number").intValue(), params.get("GA_mutate_number").intValue(), gap);

			double start = System.nanoTime();
			for (int i = 1; i <= params.get("iterations").intValue(); i++) {
				if (i%10 == 0){
				ga.train();
				results += i + ",";
				results += ef.value(ga.getOptimal()) + ",";
				results += (System.nanoTime() - start) / Math.pow(10, 9) + ",";
				results += params.get("GA_population").intValue() + ",";
				results += params.get("GA_mate_number").intValue() + ",";
				results += params.get("GA_mutate_number").intValue() + "\n";
				}
			}

			Utils.writeOutputToFile(outputDir,
					"tspGA_POP" 
			+ params.get("GA_population").intValue() + "_MAT"
							+ params.get("GA_mate_number").intValue() + "_MUT"
							+ params.get("GA_mutate_number").intValue() 
							+ ".csv",
					results);
			
			//Uncomment if wanting to varying the GA parameters
			/*Utils.writeOutputToFile(outputDir,
					"tspGA_POP" + ".csv",
					results);*/

			System.out.println("  GA terminated run " + params.get("run").intValue());
		}

		private void tspOptimizationMIMIC() {
			String results = "";


			Distribution df = new DiscreteDependencyTree(.1, ranges);
			ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

			MIMIC mimic = new MIMIC(params.get("MIMIC_samples").intValue(), params.get("MIMIC_to_keep").intValue(),
					pop);

			double start = System.nanoTime();
			for (int i = 1; i <= params.get("iterations").intValue(); i++) {
				if (i%10 == 0){
				mimic.train();
				results += i + ",";
				results += ef.value(mimic.getOptimal()) + ",";
				results += (System.nanoTime() - start) / Math.pow(10, 9) + ",";
				results += params.get("MIMIC_samples").intValue() + ",";
				results += params.get("MIMIC_to_keep").intValue() + "\n";
				}
			}

			Utils.writeOutputToFile(outputDir, "tspMIMIC_SAM" + params.get("MIMIC_samples").intValue() + "_KEP"
					+ params.get("MIMIC_to_keep").intValue() + ".csv", results);
			System.out.println("  MIMIC terminated run " + params.get("run").intValue());
		}

		@Override
		public void run() {
			switch (algorithm) {
			case RHC:
				tspOptimizationRHC();
				break;
			case SA:
				tspOptimizationSA();
				break;
			case GA:
				tspOptimizationGA();
				break;
			case MIMIC:
				tspOptimizationMIMIC();
				break;
			}
		}

	}

	public static void main(String[] args) throws InterruptedException {
		ArrayList<Thread> threads = new ArrayList<Thread>();
		HashMap<String, Double> params = new HashMap<String, Double>();
		int runs = 10;

		for (int i = 1; i <= runs; i++) {
			params.put("run", (double) i);

			params.put("iterations", 5000.);
			threads.add(new Thread(new tspOptimization(Algorithm.RHC, new HashMap<String, Double>(params))));

			params.put("SA_initial_temperature", 1E12);
			params.put("SA_cooling_factor", .95);
			params.put("iterations", 5000.);
			threads.add(new Thread(new tspOptimization(Algorithm.SA, new HashMap<String, Double>(params))));

			// Uncomment if wanting to vary the GA parameters
	        /*double [] toMate = {50., 50., 50. , 100., 100., 100., 150., 150., 150.}; 
	        double [] toMutate = {20., 40., 60., 20., 40., 60., 20., 40., 60.}; 
	        for (int k = 0; k < toMate.length; k ++){
			params.put("GA_population", 200.);
			params.put("GA_mate_number", toMate[k]);
			params.put("GA_mutate_number", toMutate[k]);
			params.put("iterations", 5000.);
			threads.add(new Thread(new tspOptimization(Algorithm.GA, new HashMap<String, Double>(params))));
	        }*/
	        
	        params.put("GA_population", 200.);
			params.put("GA_mate_number", 100.);
			params.put("GA_mutate_number", 40.);
			params.put("iterations", 5000.);
			threads.add(new Thread(new tspOptimization(Algorithm.GA, new HashMap<String, Double>(params))));
			
			params.put("MIMIC_samples", 200.);
			params.put("MIMIC_to_keep", 100.);
			params.put("iterations", 1000.);
			threads.add(new Thread(new tspOptimization(Algorithm.MIMIC, new HashMap<String, Double>(params))));
		}

		System.out.println("Start to compute TSP Problems");
		for (Thread t : threads) {
			t.start();
		}
		for (Thread t : threads) {
			t.join();
		}
		System.out.println("TSP Problems computation terminated");
	}
}