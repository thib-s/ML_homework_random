package opt.test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Random;
import java.nio.file.*;
import java.util.concurrent.ConcurrentHashMap;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SingleCrossOver;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.NQueensFitnessFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;



class Analyze_Optimization_Test_SA implements Runnable {

    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
         
    private Thread t;

    private String problem;
    private String algorithm;
    private double cooling;
    private double temperature;
    private int iterations;
    private HashMap<String, Double> params;
    private int N;
    private int T;
    private ConcurrentHashMap<String, String> other_params;
    private int run;

    Analyze_Optimization_Test_SA(
            String problem,
            String algorithm,
            double cooling,
            double temperature,
            int iterations,
            HashMap<String, Double> params,
            int N,
            int T,
            ConcurrentHashMap<String, String> other_params,
            int run
        ) {
        this.problem = problem;
        this.algorithm = algorithm;
        this.cooling = cooling;
        this.temperature = temperature;
        this.iterations = iterations;
        this.params = params;
        this.N = N;
        this.T = T;
        this.other_params = other_params;
        this.run = run;
    }

    private void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
            if (final_result) {
                String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
                String full_path = augmented_output_dir + "/" + file_name;
                Path p = Paths.get(full_path);
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized (pwtr) {
                    pwtr.println(results);
                    pwtr.close();
                }
            }
            else {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }
    
    private double[][] buildPoints(int N) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        return points;
    }
    
    private int[] buildCopies(int N) {
        int[] copies = new int[N];
        Arrays.fill(copies, COPIES_EACH);
        return copies;
    }
    
    private double[] buildWeights(int N) {
	    Random random = new Random();
        double[] weights = new double[N];
        for (int i = 0; i < N; i++) {
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        return weights;
    }
    
    private double[] buildVolumes(int N) {
	    Random random = new Random();
        double[] volumes = new double[N];
        for (int i = 0; i < N; i++) {
            volumes[i] = random.nextDouble() * MAX_VOLUME;
        }
        return volumes;
    }

    public void run() {
        try {
            EvaluationFunction ef = null;
            Distribution odd = null;
            NeighborFunction nf = null;
            MutationFunction mf = null;
            CrossoverFunction cf = null;
            Distribution df = null;
            int[] ranges;
            double knapsackVolume = MAX_VOLUME * N * COPIES_EACH * .4;
            
            switch (this.problem) {
                case "four_peaks":
                    ranges = new int[N];
                    Arrays.fill(ranges, 2);
                    ef = new FourPeaksEvaluationFunction(T);
                    odd = new DiscreteUniformDistribution(ranges);
                    nf = new DiscreteChangeOneNeighbor(ranges);
                    mf = new DiscreteChangeOneMutation(ranges);
                    cf = new SingleCrossOver();
                    df = new DiscreteDependencyTree(.1, ranges);
                    break;
            }

            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            String results = "";
            double optimal_value = -1;
            double start = System.nanoTime();

            switch (this.algorithm) {
                case "SA":
                    SimulatedAnnealing sa = new SimulatedAnnealing(
                            params.get("SA_initial_temperature"),
                            params.get("SA_cooling_factor"),
                            hcp
                    );
                    for (int i = 0; i <= this.iterations; i++) {
                        results += sa.train() + "\n";
                    }
                    optimal_value = ef.value(sa.getOptimal());
                    for (int i = 0; i < this.N; i++) {
	                    results += sa.getOptimal().getData().get(i) + ",";
	                }
	                results += "\n";
                    break;
            }
            
            double end = System.nanoTime();
            double timeSeconds = (end - start) / Math.pow(10,9);
            
            results += "\n" +
                    "Problem: " + this.problem + "\n" +
                    "Algorithm: " + this.algorithm + "\n" +
                    "Cooling: " + this.cooling + "\n" +
                    "Temperature" + this.temperature + "\n" +
                    "Num Items: " + this.N + "\n" +
                    "Optimal Value: " + optimal_value + "\n" +
                    "Time: " + timeSeconds + "s\n";
            String final_result = "";
            final_result =
                    this.problem + "," +
                    this.algorithm + "," +
                    this.cooling + "," +
                    this.temperature + "," +
                    this.N + "," +
                    this.iterations + "," +
                    this.run + "," +
                    timeSeconds + "," +
                    optimal_value;
            write_output_to_file(this.other_params.get("output_folder"), "results_SA_cooling.csv", final_result, true);
            String file_name =
                    this.problem + "_" + this.algorithm + this.cooling + this.temperature + "_N_" + this.N +
                    "_iter_" + this.iterations + "_run_" + this.run + ".csv";
            write_output_to_file(this.other_params.get("output_folder"), file_name, results, false);
            System.out.println(results);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void start () {
        if (t == null)
        {
            t = new Thread (this);
            t.start ();
        }
    }
}


public class OptimizationTestSA {

    public static void main(String[] args) {

        ConcurrentHashMap<String, String> other_params = new ConcurrentHashMap<>();
        other_params.put("output_folder","Optimization_Results");
        int num_runs = 10;
        double [] cooling = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95};
        double [] temperature = {1e2, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9}; 
        for (int k = 0; k < cooling.length; k ++){
        //Four Peaks Test
        HashMap<String, Double> four_peaks_test_params = new HashMap<>();
        four_peaks_test_params.put("SA_initial_temperature",temperature[1]);
        four_peaks_test_params.put("SA_cooling_factor",cooling[k]);


        int[] N = {50};
        int[] iterations = {20000};
        String[] algorithms = {"SA"};
        for (int i = 0; i < algorithms.length; i++) {
            for (int j = 0; j < N.length; j++) {
				for (int l = 0; l < num_runs; l++) {
					new Analyze_Optimization_Test_SA(
							"four_peaks",
							algorithms[i],
							cooling[k],
							temperature[i],
							iterations[i],
							four_peaks_test_params,
							N[j],
							N[j]/5, 
							other_params,
							l
					).start();
				}
            }
        }
       }
    }
}