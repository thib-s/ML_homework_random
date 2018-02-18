//username : mzr3
package opt.test;

import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.NetworkTrainer;
import func.nn.NeuralNetwork;
import func.nn.backprop.*;
import func.nn.backprop.BatchBackPropagationTrainer;


//Java imports
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.io.*;
import java.text.*;
import java.lang.Runnable;
import java.lang.Thread;
import java.lang.reflect.Array;

public class DiabetesTest {
	private static String outputDir = "./DiabetesResults";
	private Thread t;
	private String threadName;
	private String optimizationAlgorithm;

	private String training_data_file;
	private String output_dir;
	private Instance[] training_instances;
	private static Instance[] test_instances;
	private DataSet training_set;
	private String comments;

	private int trainingIterations;

	//ANN specifications
	private static int inputLayer;
	private static int outputLayer;
	private static int hiddenLayer;


	private DecimalFormat df = new DecimalFormat("0.000");

	private static enum Algorithm {
		RHC, SA, GA,BP
	};

	private static class DiabetesOptimization implements Runnable {
		
		private double[] calculate_accuracy(Instance[] instances, Instance optimalInstance) {
			int correct = 0, incorrect = 0;
			BackPropagationNetwork network = new BackPropagationNetworkFactory().createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});
			network.setWeights(optimalInstance.getData());
			for(int j = 0; j < instances.length; j++) {
				network.setInputValues(instances[j].getData());
				network.run();

				double predicted = Double.parseDouble(instances[j].getLabel().toString());
				double actual = Double.parseDouble(network.getOutputValues().toString());

				double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
			}
			double[] A = {correct,incorrect,correct*100.0/(correct+incorrect)};
			return A;
		}
		private double[] calculate_accuracy(Instance[] instances, NeuralNetwork net) {
			int correct = 0, incorrect = 0;
			BackPropagationNetwork network = new BackPropagationNetworkFactory()
					.createClassificationNetwork(new int[] { inputLayer, hiddenLayer, outputLayer });
			network.setWeights(net.getWeights());
			for (int j = 0; j < instances.length; j++) {
				network.setInputValues(instances[j].getData());
				network.run();
				double predicted = Double.parseDouble(instances[j].getLabel().toString());
				double actual = Double.parseDouble(network.getOutputValues().toString());
				@SuppressWarnings("unused")
				double temp = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
			}
			double[] A = {correct,incorrect,correct*100.0/(correct+incorrect)};
			return A;
		}
		
		
		private  GradientErrorMeasure measure = new SumOfSquaresError();
		private double[] convert_to_double_arr(String[] str_array) {
			double[] double_arr = new double[str_array.length];
			for (int i=0; i < str_array.length; i++) {
				double_arr[i] = Double.parseDouble(str_array[i]);
			}
			return double_arr;
		}
		private Instance[] initializeInstances(String data_file) {

			Instance[] instances = null;

			try {
				ArrayList<String []> instance_list = new ArrayList();
				String line;
				BufferedReader br = new BufferedReader(new FileReader(new File(data_file)));
				while ((line = br.readLine()) != null) {
					instance_list.add(line.split(","));
				}

				instances = new Instance[instance_list.size()];

				for(int i = 0; i < instances.length; i++) {
					double[] attributes = convert_to_double_arr(instance_list.get(i));
					instances[i] = new Instance(Arrays.copyOfRange(attributes, 0, attributes.length - 1)); // Create an instance with the attributes
					instances[i].setLabel(new Instance(attributes[attributes.length - 1])); // Set the label for each instance
				}
			}
			catch (Exception e) {
				e.printStackTrace();
			}
			return instances;
		}
		//Prepare instances and dataset
		private Instance[] training_instances ;
		private DataSet training_set ;
		private Instance [] test_instances;



		public DiabetesOptimization(Algorithm algorithm, HashMap<String, Double> params) {
			training_instances = initializeInstances(".ABAGAIL/src/opt/test/diabetes_train.csv");
			test_instances = initializeInstances("./ABAGAIL/src/opt/test/diabetes_test.csv");
			training_set = new DataSet(training_instances);
			this.algorithm = algorithm;
			this.params = params;
			inputLayer = training_instances[0].size() - 1;
			outputLayer = 1;
			hiddenLayer = (int)(inputLayer + outputLayer)/2;
		}
		//ANN specifications
		private int inputLayer ;
		private int outputLayer ;
		private int hiddenLayer;
		// TODO Auto-generated constructor stub
		private Algorithm algorithm;
		private DecimalFormat df = new DecimalFormat("0.000");
		private HashMap<String, Double> params;
		private void DiabetesOptimizationRHC() {
			String results = "";
			BackPropagationNetwork network = new BackPropagationNetworkFactory().createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});
			NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(training_set, network, measure);
			OptimizationAlgorithm oa =new RandomizedHillClimbing(nnop);

			double time_training=0;
			for(int k = 1; k <= params.get("iterations").intValue(); k++) {
				results += k + ",";
				results +=  params.get("run").intValue()+ ",";
				double start = System.nanoTime();
				double error = 1/oa.train();
				double end = System.nanoTime();
				time_training += (end - start)/Math.pow(10,9);
				results += df.format(error) + ",";
				Instance optimalInstance = oa.getOptimal();
				double[] training_accuracy = calculate_accuracy(training_instances, optimalInstance) ;
				String pourcentage_training = df.format(training_accuracy[2]/100.0f);
				double[] test_accuracy = calculate_accuracy(test_instances, optimalInstance);
				String correct_test=df.format(test_accuracy[0]);
				String incorrect_test=df.format(test_accuracy[1]);
				String pourcentage_test=df.format(test_accuracy[2]/100.0f);
				String trainingTime = df.format(time_training);
				results += pourcentage_training + ",";
				results += trainingTime + ",";
				results += correct_test + ",";
				results += incorrect_test + ",";
				results += pourcentage_test + "\n";
			}


			Utils.writeOutputToFile(outputDir, "DiabetesRHC.csv", results);
			System.out.println("  RHC terminated run " + params.get("run").intValue());
		}

		private void DiabetesOptimizationSA() {
			String results = "";
			BackPropagationNetwork network = new BackPropagationNetworkFactory().createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});
			NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(training_set, network, measure);
			OptimizationAlgorithm oa =new SimulatedAnnealing(1E11, .95, nnop);

			double time_training=0;
			for(int k = 1; k <= params.get("iterations").intValue(); k++) {
				results += k + ",";
				results +=  params.get("run").intValue()+ ",";
				double start = System.nanoTime();
				double error = 1/oa.train();
				double end = System.nanoTime();
				time_training += (end - start)/Math.pow(10,9);
				results += df.format(error) + ",";
				Instance optimalInstance = oa.getOptimal();
				double[] training_accuracy = calculate_accuracy(training_instances, optimalInstance) ;
				String pourcentage_training = df.format(training_accuracy[2]/100.0f);
				double[] test_accuracy = calculate_accuracy(test_instances, optimalInstance);
				String correct_test=df.format(test_accuracy[0]);
				String incorrect_test=df.format(test_accuracy[1]);
				String pourcentage_test=df.format(test_accuracy[2]/100.0f);
				String trainingTime = df.format(time_training);
				results += pourcentage_training + ",";
				results += trainingTime + ",";
				results += correct_test + ",";
				results += incorrect_test + ",";
				results += pourcentage_test + ",";
				results += params.get("SA_cooling_factor") + ",";
				results += params.get("SA_initial_temperature").intValue() + "\n";
			}


			Utils.writeOutputToFile(outputDir, "DiabetesSA_IT" 
			+ params.get("SA_initial_temperature") + "_CF"
					+ params.get("SA_cooling_factor") 
					+ ".csv", results);
			// Uncomment if wanting to vary the SA parameters
			/*Utils.writeOutputToFile(outputDir, "DiabetesSA_IT" 
							+ ".csv", results);*/
			System.out.println("  SA terminated run " + params.get("run").intValue());
		}
		

		private void DiabetesOptimizationGA() {
			String results = "";
			BackPropagationNetwork network = new BackPropagationNetworkFactory().createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});
			NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(training_set, network, measure);
			OptimizationAlgorithm oa =new RandomizedHillClimbing(nnop);

			double time_training=0;
			for(int k = 1; k <= params.get("iterations").intValue(); k++) {
				results += k + ",";
				results +=  params.get("run").intValue()+ ",";
				double start = System.nanoTime();
				double error = 1/oa.train();
				double end = System.nanoTime();
				time_training += (end - start)/Math.pow(10,9);
				results += df.format(error) + ",";
				Instance optimalInstance = oa.getOptimal();
				double[] training_accuracy = calculate_accuracy(training_instances, optimalInstance) ;
				String pourcentage_training = df.format(training_accuracy[2]/100.0f);
				double[] test_accuracy = calculate_accuracy(test_instances, optimalInstance);
				String correct_test=df.format(test_accuracy[0]);
				String incorrect_test=df.format(test_accuracy[1]);
				String pourcentage_test=df.format(test_accuracy[2]/100.0f);
				String trainingTime = df.format(time_training);
				results += pourcentage_training + ",";
				results += trainingTime + ",";
				results += correct_test + ",";
				results += incorrect_test + ",";
				results += pourcentage_test + ",";
				results += params.get("GA_population").intValue() + ",";
				results += params.get("GA_mate_number").intValue() + ",";
				results += params.get("GA_mutate_number").intValue() + "\n";
			}


			Utils.writeOutputToFile(outputDir,
					"DiabetesGA_POP" 
					+ params.get("GA_population").intValue() + "_MAT"
					+ params.get("GA_mate_number").intValue() + "_MUT"
					+ params.get("GA_mutate_number").intValue() 
							+ ".csv",
							results);
			// Uncomment if wanting to vary the GA parameters
			/*Utils.writeOutputToFile(outputDir,
					"DiabetesGA_POP" 
							+ ".csv",
							results);*/
			
			System.out.println("  GA terminated run " + params.get("run").intValue());
		}
		private void DiabetesOptimizationBP() {
			String results = "";

			BackPropagationNetwork network = new BackPropagationNetworkFactory().createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});
			NetworkTrainer trainer = new BatchBackPropagationTrainer(training_set, network, measure,
					new BasicUpdateRule(params.get("learning_rate")));
			double time_training = 0;
			for (int k = 1; k <= params.get("iterations").intValue(); k++) {
				results += k + ",";
				results +=  params.get("run").intValue()+ ",";
				double start = System.nanoTime();
				double error=trainer.train();
				double end = System.nanoTime();
				time_training += (end - start)/Math.pow(10,9);
				results += df.format(error) + ",";
				double[] training_accuracy = calculate_accuracy(training_instances, trainer.getNetwork()) ;
				String pourcentage_training = df.format(training_accuracy[2]/100.0f);
				double[] test_accuracy = calculate_accuracy(test_instances, trainer.getNetwork());
				String correct_test=df.format(test_accuracy[0]);
				String incorrect_test=df.format(test_accuracy[1]);
				String pourcentage_test=df.format(test_accuracy[2]/100.0f);
				String trainingTime = df.format(time_training);
				results += pourcentage_training + ",";
				results += trainingTime + ",";
				results += correct_test + ",";
				results += incorrect_test + ",";
				results += pourcentage_test + "\n";
			}

			Utils.writeOutputToFile(outputDir, "NeuralNetworkBP.csv", results);
			System.out.println("  BP terminated run " + params.get("run").intValue());
		}
		public void run() {
			switch (algorithm) {
			case RHC:
				DiabetesOptimizationRHC();
				break;
			case SA:
				DiabetesOptimizationSA();
				break;
			case GA:
				DiabetesOptimizationGA();
				break;
			case BP:
				DiabetesOptimizationBP();
			}
		}
	}
	public static void main(String[] args) throws InterruptedException {
		Locale.setDefault(new Locale("en", "US"));
		ArrayList<Thread> threads = new ArrayList<Thread>();
		HashMap<String, Double> params = new HashMap<String, Double>();
		int runs = 10;

		for (int i = 1; i <= runs; i++) {
			params.put("run", (double) i);

			params.put("iterations", 100.);
			threads.add(new Thread(new DiabetesOptimization(Algorithm.RHC, new HashMap<String, Double>(params))));
	        
			//Uncomment if wanting to vary SA parameters
			/*double [] cooling = {0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90};
	        double [] temperature = {1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9}; 
	        for (int k=0; k < cooling.length; k++){
	        	for (int j= 0; j < temperature.length;j++){
				params.put("SA_initial_temperature", temperature[j]);
				params.put("SA_cooling_factor", cooling[k]);
				params.put("iterations", 100.);
				threads.add(new Thread(new DiabetesOptimization(Algorithm.SA, new HashMap<String, Double>(params))));
	        }
		}*/
	        
	        params.put("SA_initial_temperature", 100.);
			params.put("SA_cooling_factor", 0.95);
			params.put("iterations", 100.);
			threads.add(new Thread(new DiabetesOptimization(Algorithm.SA, new HashMap<String, Double>(params))));
			
			//Uncomment if wanting to vary GA parameters
	        /*double [] populationSize = {100.,100.,200.,200.,500.};
	        double [] toMate = {50., 25., 75. , 125., 150.}; 
	        double [] toMutate = {20.,10.,50.,60.,100.};
	        for (int l = 0; l < toMutate.length;l++){
	        	for (int k = 0; k < toMate.length; k ++){
					params.put("GA_population", populationSize[k]);
					params.put("GA_mate_number", toMate[k]);
					params.put("GA_mutate_number", toMutate[k]);
					params.put("iterations", 100.);
					threads.add(new Thread(new DiabetesOptimization(Algorithm.GA, new HashMap<String, Double>(params))));
	        }
	        	}*/
	        
	        params.put("GA_population", 200.);
			params.put("GA_mate_number", 100.);
			params.put("GA_mutate_number", 40.);
			params.put("iterations", 100.);
			threads.add(new Thread(new DiabetesOptimization(Algorithm.GA, new HashMap<String, Double>(params))));
			
			params.put("learning_rate", .0001);
			params.put("iterations", 100.);
			threads.add(new Thread(new DiabetesOptimization(Algorithm.BP,
					new HashMap<String, Double>(params))));
		}

		System.out.println("Start to compute ANN optimisation");
		for (Thread t : threads) {
			t.start();
		}
		for (Thread t : threads) {
			t.join();
		}
		System.out.println(" ANN optimisation computation terminated");
	}

}



