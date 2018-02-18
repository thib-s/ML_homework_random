package opt.test;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.SingleCrossOver;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;


public class FourPeaksProblem {

	private static String outputDir = "./OptimizationResults";

	/** The N value */
	private static final int N = 200;

	/** The T value */
	private static final int T = N / 5;

	/** The vector of points */
	private static int[] ranges = new int[N];
	static {
		Arrays.fill(ranges, 2);
	}

	/** The evaluation function */
	private static EvaluationFunction ef = new FourPeaksEvaluationFunction(T);

	private static Distribution odd = new DiscreteUniformDistribution(ranges);

	private static enum Algorithm {
		RHC, SA, GA, MIMIC
	};

	/**
	 * Four Peaks Problem optimizations using different algorithms
	 */
	private static class FourPeaksOptimization implements Runnable {

		private Algorithm algorithm;

		private HashMap<String, Double> params;

		public FourPeaksOptimization(Algorithm algorithm, HashMap<String, Double> params) {
			this.algorithm = algorithm;
			this.params = params;
		}

		private void FourPeaksOptimizationRHC() {
			String results = "";


			NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
			HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

			RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);

			double start = System.nanoTime();
			for (int i = 1; i <= params.get("iterations").intValue(); i++) {
				if (i%10 == 0){
				rhc.train();
				results += i + ";";
				results += ef.value(rhc.getOptimal()) + ";";
				results += (System.nanoTime() - start) / Math.pow(10, 9) + "\n";
			}
			}

			Utils.writeOutputToFile(outputDir, "FourPeaksRHC.csv", results);
			System.out.println("  RHC terminated run " + params.get("run").intValue());
		}

		private void FourPeaksOptimizationSA() {
			String results = "";


			NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
			HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

			SimulatedAnnealing sa = new SimulatedAnnealing(params.get("SA_initial_temperature"),
					params.get("SA_cooling_factor"), hcp);

			double start = System.nanoTime();
			for (int i = 1; i <= params.get("iterations").intValue(); i++) {
				if (i%10 == 0){
				sa.train();
				results += i + ";";
				results += ef.value(sa.getOptimal()) + ";";
				results += (System.nanoTime() - start) / Math.pow(10, 9) + ";";
				results += params.get("SA_initial_temperature") + ";";
				results += params.get("SA_cooling_factor") + "\n";
				}
			}

			Utils.writeOutputToFile(outputDir, "FourPeaksSA_cooling"  + ".csv", results);
			System.out.println("  SA terminated run " + params.get("run").intValue());
		}

		private void FourPeaksOptimizationGA() {
			String results = "";


			MutationFunction mf = new DiscreteChangeOneMutation(ranges);
			CrossoverFunction cf = new SingleCrossOver();
			GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

			StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(params.get("GA_population").intValue(),
					params.get("GA_mate_number").intValue(), params.get("GA_mutate_number").intValue(), gap);

			double start = System.nanoTime();
			for (int i = 1; i <= params.get("iterations").intValue(); i++) {
				if (i % 10 == 0){
				ga.train();
				results += i + ";";
				results += ef.value(ga.getOptimal()) + ";";
				results += (System.nanoTime() - start) / Math.pow(10, 9) + ";";
				results += params.get("GA_population").intValue() + ";";
				results += params.get("GA_mate_number").intValue() + ";";
				results += params.get("GA_mutate_number").intValue() + "\n";
				}
			}

			Utils.writeOutputToFile(outputDir,
					"FourPeaksGA_POP" + params.get("GA_population").intValue() + "_MAT"
							+ params.get("GA_mate_number").intValue() + "_MUT"
							+ params.get("GA_mutate_number").intValue() + ".csv",
					results);
			System.out.println("  GA terminated run " + params.get("run").intValue());
		}

		private void FourPeaksOptimizationMIMIC() {
			String results = "";

			Distribution df = new DiscreteDependencyTree(.1, ranges);
			ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

			MIMIC mimic = new MIMIC(params.get("MIMIC_samples").intValue(), params.get("MIMIC_to_keep").intValue(),
					pop);

			double start = System.nanoTime();
			for (int i = 1; i <= params.get("iterations").intValue(); i++) {
				if (i%10 == 0){
				mimic.train();
				results += i + ";";
				results += ef.value(mimic.getOptimal()) + ";";
				results += (System.nanoTime() - start) / Math.pow(10, 9) + ";";
				results += params.get("MIMIC_samples").intValue() + ";";
				results += params.get("MIMIC_to_keep").intValue() + "\n";
			}
			}

			Utils.writeOutputToFile(outputDir, "FourPeaksMIMIC_SAM" + params.get("MIMIC_samples").intValue() + "_KEP"
					+ params.get("MIMIC_to_keep").intValue() + ".csv", results);
			System.out.println("  MIMIC terminated run " + params.get("run").intValue());
		}

		@Override
		public void run() {
			switch (algorithm) {
			case RHC:
				FourPeaksOptimizationRHC();
				break;
			case SA:
				FourPeaksOptimizationSA();
				break;
			case GA:
				FourPeaksOptimizationGA();
				break;
			case MIMIC:
				FourPeaksOptimizationMIMIC();
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
			//threads.add(new Thread(new FourPeaksOptimization(Algorithm.RHC, new HashMap<String, Double>(params))));
	        double [] cooling = {0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90};
	        double [] temperature = {1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9}; 
	        for (int k=0; k < cooling.length; k++){
			params.put("SA_initial_temperature", 100.);
			params.put("SA_cooling_factor", cooling[k]);
			params.put("iterations", 5000.);
			threads.add(new Thread(new FourPeaksOptimization(Algorithm.SA, new HashMap<String, Double>(params))));
	        }
			params.put("GA_population", 100.);
			params.put("GA_mate_number", 50.);
			params.put("GA_mutate_number", 10.);
			params.put("iterations", 5000.);
			//threads.add(new Thread(new FourPeaksOptimization(Algorithm.GA, new HashMap<String, Double>(params))));

			params.put("MIMIC_samples", 50.);
			params.put("MIMIC_to_keep", 10.);
			params.put("iterations", 1000.);
			//threads.add(new Thread(new FourPeaksOptimization(Algorithm.MIMIC, new HashMap<String, Double>(params))));
		
		}

		System.out.println("Start to compute Four Peaks Problems");
		for (Thread t : threads) {
			t.start();
		}
		for (Thread t : threads) {
			t.join();
		}
		System.out.println("Four Peaks Problems computation terminated");
	}
}
