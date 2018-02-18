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



class Analyze_Optimization_Test_GA implements Runnable {

    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
         
    private Thread t;

    private String problem;
    private String algorithm;
    private int populationSize;
    private int toMate;
    private int toMutate;
    private int iterations;
    private HashMap<String, Integer> params;
    private int N;
    private int T;
    private ConcurrentHashMap<String, String> other_params;
    private int run;

    Analyze_Optimization_Test_GA(
            String problem,
            String algorithm,
            int populationSize,
            int toMate,
            int toMutate,
            int iterations,
            HashMap<String,Integer> params,
            int N,
            int T,
            ConcurrentHashMap<String, String> other_params,
            int run
        ) {
        this.problem = problem;
        this.algorithm = algorithm;
        this.populationSize = populationSize;
        this.toMate = toMate;
        this.toMutate = toMutate;
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
                case "tsp":
					ranges = new int[N];
					Arrays.fill(ranges, N);
					ef = new TravelingSalesmanRouteEvaluationFunction(buildPoints(N));
					odd = new DiscretePermutationDistribution(N);
					nf = new SwapNeighbor();
					mf = new SwapMutation(); // Does the same thing
					cf = new TravelingSalesmanCrossOver((TravelingSalesmanEvaluationFunction) ef);
					df = new DiscreteDependencyTree(.1, ranges);
            }

            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            String results = "";
            double optimal_value = -1;
            double start = System.nanoTime();

            switch (this.algorithm) {
                case "GA":
                    StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(
                            params.get("GA_population").intValue(),
                            params.get("GA_mate_number").intValue(),
                            params.get("GA_mutate_number").intValue(),
                            gap
                    );
                    for (int i = 0; i <= this.iterations; i++) {
                        results += ga.train() + "\n";
                    }
                    optimal_value = ef.value(ga.getOptimal());
                    for (int i = 0; i < this.N; i++) {
	                    results += ga.getOptimal().getData().get(i) + ",";
	                }
	                results += "\n";
                    break;
            }
            
            double end = System.nanoTime();
            double timeSeconds = (end - start) / Math.pow(10,9);
            
            results += "\n" +
                    "Problem: " + this.problem + "\n" +
                    "Algorithm: " + this.algorithm + "\n" +
                    "PopulationSize : " + this.populationSize + "\n" +
                    "toMate : " + this.toMate + "\n" +
                    "toMutate : " + this.toMutate + "\n" +
                    "Num Items: " + this.N + "\n" +
                    "Iterations : " + this.iterations + "\n" +
                    "Optimal Value: " + optimal_value + "\n" +
                    "Time: " + timeSeconds + "s\n";
            String final_result = "";
            final_result =
                    this.problem + "," +
                    this.algorithm + "," +
                    this.populationSize + "," +
                    this.toMate + "," +
                    this.toMutate + "," +
                    this.N + "," +
                    this.iterations + "," +
                    this.run + "," +
                    timeSeconds + "," +
                    optimal_value;
            write_output_to_file(this.other_params.get("output_folder"), "results_GA.csv", final_result, true);
            String file_name =
                    this.problem + "_" + this.algorithm + this.populationSize + this.toMate + this.toMutate + "_N_" + this.N +
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


public class OptimizationTestGA {

    public static void main(String[] args) {

        ConcurrentHashMap<String, String> other_params = new ConcurrentHashMap<>();
        other_params.put("output_folder","Optimization_Results");
        int num_runs = 10;

        int [] populationSize = {100, 100, 100 , 100, 100, 100, 100, 100, 100,
        						 200, 200, 200 , 200, 200, 200, 200, 200, 200};
        int [] toMate = {25, 25, 25 , 50, 50, 50, 75, 75, 75,
        				50, 50, 50 , 100, 100, 100, 150, 150, 150}; 
        int [] toMutate = {10, 20, 30, 10, 20, 30, 10, 20, 30,
        				   20, 40, 60, 20, 40, 60, 20, 40, 60}; 
        for (int k = 0; k < toMate.length; k ++){

        //Traveling Salesman Problem
        HashMap<String, Integer> tsp_test_params = new HashMap<>();
        tsp_test_params.put("GA_population",populationSize[k]);
        tsp_test_params.put("GA_mate_number",toMate[k]);
        tsp_test_params.put("GA_mutate_number",toMutate[k]);


        int[] N = {50};
        String [] algorithms = {"GA"};
        int[] iterations = {1000};
        for (int i = 0; i < algorithms.length; i++) {
            for (int j = 0; j < N.length; j++) {
				for (int l = 0; l < num_runs; l++) {
					new Analyze_Optimization_Test_GA(
							"tsp",
							algorithms[i],
							populationSize[k],
							toMate[k],
							toMutate[k],
							iterations[i],
							tsp_test_params,
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