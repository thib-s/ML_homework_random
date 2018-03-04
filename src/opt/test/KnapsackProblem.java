//username : mzr3
package opt.test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

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
import opt.example.KnapsackEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;


public class KnapsackProblem {

    private static String outputDir = "./OptimizationResults";

    /**
     * Random number generator
     */
    private static final Random random = new Random();

    /**
     * The number of items
     */
    private static final int NUM_ITEMS = 40;

    /**
     * The number of copies each
     */
    private static final int COPIES_EACH = 4;

    /**
     * The maximum value for a single element
     */
    private static final double MAX_VALUE = 50;

    /**
     * The maximum weight for a single element
     */
    private static final double MAX_WEIGHT = 50;

    /**
     * The maximum weight for the knapsack
     */
    private static final double MAX_KNAPSACK_WEIGHT = MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;

    /**
     * create and initialize the copies of the items
     */
    private static int[] copies = new int[NUM_ITEMS];

    static {
        Arrays.fill(copies, COPIES_EACH);
    }

    /**
     * create and initialize the values and weights of the items
     */
    private static double[] values = new double[NUM_ITEMS];
    private static double[] weights = new double[NUM_ITEMS];

    static {
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
    }

    /**
     * create and initialize the space of the knapsack
     */
    private static int[] ranges = new int[NUM_ITEMS];

    static {
        Arrays.fill(ranges, COPIES_EACH + 1);
    }

    /**
     * The evaluation function
     */
    private static EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);

    private static Distribution odd = new DiscreteUniformDistribution(ranges);

    private static enum Algorithm {
        RHC, SA, GA, MIMIC
    }

    ;

    /**
     * Knapsack Problem optimizations using different algorithms
     */
    private static class KnapsackOptimization implements Runnable {

        private Algorithm algorithm;

        private HashMap<String, Double> params;

        public KnapsackOptimization(Algorithm algorithm, HashMap<String, Double> params) {
            this.algorithm = algorithm;
            this.params = params;
        }

        private void KnapsackOptimizationRHC() {
            String results = "";
            String header = "iter, value, time\n";

            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);

            double start = System.nanoTime();
            for (int i = 1; i <= params.get("iterations").intValue(); i++) {
                rhc.train();
                if (i % 10 == 0) {
                    results += i + ";";
                    results += ef.value(rhc.getOptimal()) + ";";
                    results += (System.nanoTime() - start) / Math.pow(10, 9) + "\n";
                }
            }

            Utils.writeOutputToFile(outputDir, "KnapsackRHC.csv", results, header);
            System.out.println("  RHC terminated run " + params.get("run").intValue());
        }

        private void KnapsackOptimizationSA() {
            String results = "";
            String header = "iter, value, time, SA_initial_temperature, SA_cooling_factor\n";

            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

            SimulatedAnnealing sa = new SimulatedAnnealing(params.get("SA_initial_temperature"),
                    params.get("SA_cooling_factor"), hcp);

            double start = System.nanoTime();
            for (int i = 1; i <= params.get("iterations").intValue(); i++) {
                if (i % 10 == 0) {
                    sa.train();
                    results += i + ";";
                    results += ef.value(sa.getOptimal()) + ";";
                    results += (System.nanoTime() - start) / Math.pow(10, 9) + ";";
                    results += params.get("SA_initial_temperature") + ";";
                    results += params.get("SA_cooling_factor") + "\n";
                }
            }

            Utils.writeOutputToFile(outputDir, "KnapsackSA_IT.csv", results, header);
            System.out.println("  SA terminated run " + params.get("run").intValue());
        }

        private void KnapsackOptimizationGA() {
            String results = "";
            String header = "iter, value, time, GA_population, GA_mate_number, GA_mutate_number\n";


            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(params.get("GA_population").intValue(),
                    params.get("GA_mate_number").intValue(), params.get("GA_mutate_number").intValue(), gap);

            double start = System.nanoTime();
            for (int i = 1; i <= params.get("iterations").intValue(); i++) {
                if (i % 10 == 0) {
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
                    "KnapsackGA.csv",
                    results, header);
            System.out.println("  GA terminated run " + params.get("run").intValue());
        }

        private void KnapsackOptimizationMIMIC() {
            String results = "";
            String header = "iter, value, time, MIMIC_samples, MIMIC_to_keep\n";

            Distribution df = new DiscreteDependencyTree(.1, ranges);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            MIMIC mimic = new MIMIC(params.get("MIMIC_samples").intValue(), params.get("MIMIC_to_keep").intValue(),
                    pop);

            double start = System.nanoTime();
            for (int i = 1; i <= params.get("iterations").intValue(); i++) {
                if (i % 10 == 0) {
                    mimic.train();
                    results += i + ";";
                    results += ef.value(mimic.getOptimal()) + ";";
                    results += (System.nanoTime() - start) / Math.pow(10, 9) + ";";
                    results += params.get("MIMIC_samples").intValue() + ";";
                    results += params.get("MIMIC_to_keep").intValue() + "\n";
                }
            }
//			Utils.writeOutputToFile(outputDir, "KnapsackMIMIC_tokeep"
//					+params.get("MIMIC_samples").intValue() +"_SAM"
//					+params.get("MIMIC_to_keep").intValue() +"_KP"
//					+ ".csv", results);
            Utils.writeOutputToFile(outputDir,
                    "KnapsackMIMIC" + ".csv",
                    results, header);
            System.out.println("  MIMIC terminated run " + params.get("run").intValue());
            //Uncomment if wanting to make varying samples and tokeep
            //Utils.writeOutputToFile(outputDir, "KnapsackMIMIC_tokeep" + ".csv", results);
            //System.out.println("  MIMIC terminated run " + params.get("run").intValue());
        }

        @Override
        public void run() {
            switch (algorithm) {
                case RHC:
                    KnapsackOptimizationRHC();
                    break;
                case SA:
                    KnapsackOptimizationSA();
                    break;
                case GA:
                    KnapsackOptimizationGA();
                    break;
                case MIMIC:
                    KnapsackOptimizationMIMIC();
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
            threads.add(new Thread(new KnapsackOptimization(Algorithm.RHC, new HashMap<String, Double>(params))));

            double[] initTemps = {1E12, 1E10, 1E5};
            double[] coolings = {.95, .9, .8};
            for (double initTemp : initTemps) {
                for (double cooling : coolings) {
                    params.put("SA_initial_temperature", initTemp);
                    params.put("SA_cooling_factor", cooling);
                    params.put("iterations", 5000.);
                    threads.add(new Thread(new KnapsackOptimization(Algorithm.SA, new HashMap<String, Double>(params))));
                }
            }

            double[] proportions = {0., 0.25, 0.5, 0.75, 1.};
            double[] pops = {100., 200., 300., 400.};
            for (double mateProp : proportions) {
                for (double mutateProp : proportions) {
                    for (double pop : pops) {
                        params.put("GA_population", pop);
                        params.put("GA_mate_number", mateProp * pop);
                        params.put("GA_mutate_number", mutateProp * pop);
                        params.put("iterations", 5000.);
                        threads.add(new Thread(new KnapsackOptimization(Algorithm.GA, new HashMap<String, Double>(params))));
                    }
                }
            }

            double[] samples = {100., 200., 300.};
            double[] proportions1 = {0.25, 0.5, 0.75};
            for (double sample : samples) {
                for (double prop : proportions1) {
                    params.put("MIMIC_samples", sample);
                    params.put("MIMIC_to_keep", sample * prop);
                    params.put("iterations", 1000.);
                    threads.add(new Thread(new KnapsackOptimization(Algorithm.MIMIC, new HashMap<String, Double>(params))));
                }
            }
        }

        System.out.println("Start to compute Knapsack Problems");
        for (Thread t : threads) {
            t.start();
        }
        for (Thread t : threads) {
            t.join();
        }
        System.out.println("Knapsack Problems computation terminated");
    }

}
