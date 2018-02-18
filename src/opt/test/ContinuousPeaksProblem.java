//username : mzr3
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
import opt.example.ContinuousPeaksEvaluationFunction;
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

public class ContinuousPeaksProblem {

	private static String outputDir = "./OptimizationResults";

	/** The N value */
	private static final int N = 200;

	/** The T value */
	private static final int T = N / 10;

	/** The vector of points */
	private static int[] ranges = new int[N];
	static {
		Arrays.fill(ranges, 2);
	}

	/** The evaluation function */
	private static EvaluationFunction ef = new ContinuousPeaksEvaluationFunction(T);

	private static Distribution odd = new DiscreteUniformDistribution(ranges);

	private static enum Algorithm {
		RHC, SA, GA, MIMIC
	};

	/**
	 * Continuous Peaks Problem optimizations using different algorithms
	 */
	private static class ContinuousPeaksOptimization implements Runnable {

		private Algorithm algorithm;

		private HashMap<String, Double> params;

		public ContinuousPeaksOptimization(Algorithm algorithm, HashMap<String, Double> params) {
			this.algorithm = algorithm;
			this.params = params;
		}

		private void ContinuousPeaksOptimizationRHC() {
			String results = "";


			NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
			HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

			RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);

			double start = System.nanoTime();
			for (int i = 1; i <= params.get("iterations").intValue(); i++) {
				rhc.train();
				if (i %10 == 0){
				results += i + ";";
				results += ef.value(rhc.getOptimal()) + ";";
				results += (System.nanoTime() - start) / Math.pow(10, 9) + "\n";
				}
			}

			Utils.writeOutputToFile(outputDir, "ContinuousPeaksRHC.csv", results);
			System.out.println("  RHC terminated run " + params.get("run").intValue());
		}

		private void ContinuousPeaksOptimizationSA() {
			String results = "";


			NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
			HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

			SimulatedAnnealing sa = new SimulatedAnnealing(params.get("SA_initial_temperature"),
					params.get("SA_cooling_factor"), hcp);

			double start = System.nanoTime();
			for (int i = 1; i <= params.get("iterations").intValue(); i++) {
				sa.train();
				if (i %10 == 0){
				results += i + ";";
				results += ef.value(sa.getOptimal()) + ";";
				results += (System.nanoTime() - start) / Math.pow(10, 9) + ",";
				results += params.get("SA_initial_temperature") + ",";
				results += params.get("SA_cooling_factor") + "\n";
				}
			}
			//GeneralCase
			Utils.writeOutputToFile(outputDir, "ContinuousPeaksSA_IT" + params.get("SA_initial_temperature") + "_CF"
					+ params.get("SA_cooling_factor") + ".csv", results);
			
			//When varying the temperature, uncomment
			//Utils.writeOutputToFile(outputDir,"ContinuousPeaksSA_Temperature.csv",results);
			
			//When varying the cooling factor, uncomment
			//Utils.writeOutputToFile(outputDir,"ContinuousPeaksSA_cooling.csv",results);

			System.out.println("  SA terminated run " + params.get("run").intValue());
		}

		private void ContinuousPeaksOptimizationGA() {
			String results = "";


			MutationFunction mf = new DiscreteChangeOneMutation(ranges);
			CrossoverFunction cf = new SingleCrossOver();
			GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

			StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(params.get("GA_population").intValue(),
					params.get("GA_mate_number").intValue(), params.get("GA_mutate_number").intValue(), gap);

			double start = System.nanoTime();
			for (int i = 1; i <= params.get("iterations").intValue(); i++) {
				ga.train();
				if (i %10 == 0){
				results += i + ";";
				results += ef.value(ga.getOptimal()) + ";";
				results += (System.nanoTime() - start) / Math.pow(10, 9) + ",";
				results += params.get("GA_population") + ",";
				results += params.get("GA_mate_number") + ",";
				results += params.get("GA_mutate_number") + "\n";
				}
				
			}

			Utils.writeOutputToFile(outputDir,
					"ContinuousPeaksGA_POP" + params.get("GA_population").intValue() + "_MAT"
							+ params.get("GA_mate_number").intValue() + "_MUT"
							+ params.get("GA_mutate_number").intValue() + ".csv",
							results);
			System.out.println("  GA terminated run " + params.get("run").intValue());
		}

		private void ContinuousPeaksOptimizationMIMIC() {
			String results = "";

			Distribution df = new DiscreteDependencyTree(.1, ranges);
			ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

			MIMIC mimic = new MIMIC(params.get("MIMIC_samples").intValue(), params.get("MIMIC_to_keep").intValue(),
					pop);

			double start = System.nanoTime();
			for (int i = 1; i <= params.get("iterations").intValue(); i++) {
				mimic.train();
				if (i %10 == 0){
				results += i + ";";
				results += ef.value(mimic.getOptimal()) + ";";
				results += (System.nanoTime() - start) / Math.pow(10, 9) + ",";
				results += params.get("MIMIC_samples") + ",";
				results += params.get("MIMIC_to_keep") + "\n";
				}
				
			}

			Utils.writeOutputToFile(outputDir, "ContinuousPeaksMIMIC_SAM" + params.get("MIMIC_samples").intValue() + "_KEP"
					+ params.get("MIMIC_to_keep").intValue() + ".csv", results);
			System.out.println("  MIMIC terminated run " + params.get("run").intValue());
		}

		@Override
		public void run() {
			switch (algorithm) {
			case RHC:
				ContinuousPeaksOptimizationRHC();
				break;
			case SA:
				ContinuousPeaksOptimizationSA();
				break;
			case GA:
				ContinuousPeaksOptimizationGA();
				break;
			case MIMIC:
				ContinuousPeaksOptimizationMIMIC();
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
			//Comment all but SA temperature/cooling when needed to test parameters
			//Change also needed in ContinuousPeaksOptimizationSA writing output file
			/*double[] cooling = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9};
			for (int k=0; k< cooling.length; k++){
				params.put("SA_initial_temperature", 100.);
				params.put("SA_cooling_factor", 0.95);
				params.put("iterations", 5000.);
				threads.add(new Thread(new ContinuousPeaksOptimization(Algorithm.SA, new HashMap<String, Double>(params))));
			}
			*/
			/*double[] temperature = {1E1,1E2,1E3,1E4,1E5,1E6,1E7,1E8,1E9,1E10};
			for ( int k=0; k< temperature.length; k++){
				params.put("SA_initial_temperature", temperature[k]);
				params.put("SA_cooling_factor", .95);
				params.put("iterations", 5000.);
				threads.add(new Thread(new ContinuousPeaksOptimization(Algorithm.SA, new HashMap<String, Double>(params))));
			}*/

			

			params.put("iterations", 5000.);
			threads.add(new Thread(new ContinuousPeaksOptimization(Algorithm.RHC, new HashMap<String, Double>(params))));

			params.put("SA_initial_temperature", 100.);
			params.put("SA_cooling_factor", .95);
			params.put("iterations", 5000.);
			threads.add(new Thread(new ContinuousPeaksOptimization(Algorithm.SA, new HashMap<String, Double>(params))));

			params.put("GA_population", 100.);
			params.put("GA_mate_number", 50.);
			params.put("GA_mutate_number", 10.);
			params.put("iterations", 5000.);
			threads.add(new Thread(new ContinuousPeaksOptimization(Algorithm.GA, new HashMap<String, Double>(params))));

			params.put("MIMIC_samples", 100.);
			params.put("MIMIC_to_keep", 50.);
			params.put("iterations", 1000.);
			threads.add(new Thread(new ContinuousPeaksOptimization(Algorithm.MIMIC, new HashMap<String, Double>(params))));
			
		}

		System.out.println("Start to compute Continuous Peaks Problems");
		for (Thread t : threads) {
			t.start();
		}
		for (Thread t : threads) {
			t.join();
		}
		System.out.println("Continuous Peaks Problems computation terminated");
	}
}
