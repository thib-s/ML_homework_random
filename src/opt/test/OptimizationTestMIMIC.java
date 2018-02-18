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



class Analyze_Optimization_Test_MIMIC implements Runnable {

    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
         
    private Thread t;

    private String problem;
    private String algorithm;
    private int samples;
    private int tokeep;
    private int iterations;
    private HashMap<String, Integer> params;
    private int N;
    private int T;
    private ConcurrentHashMap<String, String> other_params;
    private int run;

    Analyze_Optimization_Test_MIMIC(
            String problem,
            String algorithm,
            int samples,
            int tokeep,
            int iterations,
            HashMap<String, Integer> params,
            int N,
            int T,
            ConcurrentHashMap<String, String> other_params,
            int run
        ) {
        this.problem = problem;
        this.algorithm = algorithm;
        this.samples = samples;
        this.tokeep = tokeep;
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
                case "knapsack":
					ranges = new int[N];
        			Arrays.fill(ranges, COPIES_EACH + 1);
					ef = new KnapsackEvaluationFunction(buildWeights(N), buildVolumes(N), knapsackVolume, buildCopies(N));
					odd = new DiscreteUniformDistribution(ranges);
					nf = new DiscreteChangeOneNeighbor(ranges);
					mf = new DiscreteChangeOneMutation(ranges);
					cf = new UniformCrossOver();
					df = new DiscreteDependencyTree(.1, ranges); 
            }

            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            String results = "";
            double optimal_value = -1;
            double start = System.nanoTime();

            switch (this.algorithm) {
                case "MIMIC":
                    MIMIC mimic = new MIMIC(
                            params.get("MIMIC_samples").intValue(),
                            params.get("MIMIC_to_keep").intValue(),
                            pop
                    );
                    results = "";
                    for (int i = 0; i <= this.iterations; i++) {
                        results += mimic.train() + "\n";
                    }
                    optimal_value = ef.value(mimic.getOptimal());
                    for (int i = 0; i < this.N; i++) {
	                    results += mimic.getOptimal().getData().get(i) + ",";
	                }
	                results += "\n";
                    break;
            }
            
            double end = System.nanoTime();
            double timeSeconds = (end - start) / Math.pow(10,9);
            
            results += "\n" +
                    "Problem: " + this.problem + "\n" +
                    "Algorithm: " + this.algorithm + "\n" +
                    "Samples: " + this.samples + "\n" +
                    "Tokeep : " + this.tokeep + "\n" +
                    "Num Items: " + this.N + "\n" +
                    "Optimal Value: " + optimal_value + "\n" +
                    "Time: " + timeSeconds + "s\n";
            String final_result = "";
            final_result =
                    this.problem + "," +
                    this.algorithm + "," +
                    this.samples + "," +
                    this.tokeep + "," +
                    this.N + "," +
                    this.iterations + "," +
                    this.run + "," +
                    timeSeconds + "," +
                    optimal_value;
            write_output_to_file(this.other_params.get("output_folder"), "results_MIMIC.csv", final_result, true);
            String file_name =
                    this.problem + "_" + this.algorithm + this.samples + this.tokeep + "_N_" + this.N +
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


public class OptimizationTestMIMIC {

    public static void main(String[] args) {

        ConcurrentHashMap<String, String> other_params = new ConcurrentHashMap<>();
        other_params.put("output_folder","Optimization_Results");
        int num_runs = 10;
        int [] samples = {10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200};
        int [] tokeep = {5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}; 
        for (int k = 0; k < samples.length; k ++){
        
        //Knapsack Problem
        HashMap<String, Integer> knapsack_test_params = new HashMap<>();
        knapsack_test_params.put("MIMIC_samples",samples[k]);
        knapsack_test_params.put("MIMIC_to_keep",tokeep[k]);

        int[] N = {50};
        String[] algorithms = {"MIMIC"};
        int[] iterations = {1000};
        for (int i = 0; i < algorithms.length; i++) {
            for (int j = 0; j < N.length; j++) {
				for (int l = 0; l < num_runs; l++) {
					new Analyze_Optimization_Test_MIMIC(
							"knapsack",    
							algorithms[i],
							samples[k],
							tokeep[k],
							iterations[i],
							knapsack_test_params,
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