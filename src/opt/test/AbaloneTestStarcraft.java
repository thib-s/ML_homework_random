package opt.test;

import func.nn.Layer;
import func.nn.Link;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import util.linalg.DenseVector;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class AbaloneTestStarcraft {

    public static class OA{
        public String name;
        public OptimizationAlgorithm algorithm;
        public BackPropagationNetwork nn;
        public NeuralNetworkOptimizationProblem pb;
        public double[] errors;
        public OA(String oa_name, OptimizationAlgorithm opt, BackPropagationNetwork bpn, NeuralNetworkOptimizationProblem nnpb){
            name = oa_name;
            algorithm = opt;
            nn = bpn;
            pb = nnpb;
        }
    }

	private static String outputDir = "./OptimizationResults";
    private static int inputLayer = 72, hiddenLayer = 35, outputLayer = 200, trainingIterations = 500;
    private static Instance[] instances = initializeInstances();
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static DecimalFormat df = new DecimalFormat("0.000");

    private static String outFileParticule = "";

    public static void main(String[] args) {
        if(args.length > 1)
            outFileParticule = args[1];
        LinkedList<OA> oa_list = new LinkedList<>();

        StringBuffer results = new StringBuffer();
        BackPropagationNetwork network;
        NeuralNetworkOptimizationProblem nnop;

        network = factory.createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});
        nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
        oa_list.add(new OA("RHC",
                           new RandomizedHillClimbing(nnop),
                           network,
                           nnop));
        network = factory.createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});
        nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
        oa_list.add(new OA("SA",
                           new SimulatedAnnealing(100, .95, nnop),
                           network,
                           nnop));
        network = factory.createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});
        nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
        oa_list.add(new OA("GA",
                           new StandardGeneticAlgorithm(200, 100, 10, nnop),
                           network,
                           nnop));

        for(OA oa : oa_list) {
            double start = System.currentTimeMillis(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa); //trainer.train();
            end = System.currentTimeMillis();
            trainingTime = end - start;
            trainingTime /= 1000;

            Instance optimalInstance = oa.algorithm.getOptimal();
            oa.nn.setWeights(optimalInstance.getData());

            int predicted, actual;
            start = System.currentTimeMillis();
            for(Instance ins : instances) {
                oa.nn.setInputValues(ins.getData());
                oa.nn.run();

                actual = ins.getLabel().getData().argMax();
                predicted = oa.nn.getOutputValues().argMax();

                if(actual == predicted)
                    correct++;
                else
                    incorrect++;
            }
            end = System.currentTimeMillis();
            testingTime = end - start;
            testingTime /= 1000;

            String res = "\nResults for " + oa.name + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
            results.append(res);
        }
        Utils.writeOutputToFile(outputDir, "StarcraftTest" + outFileParticule + ".csv", results.toString());
        System.out.println(results);

        //write errors to csv file
        StringBuilder sb = new StringBuilder();
        sb.append("Opt_name, nn_shape, t, cooling, population_size, toMate, toMutate, epoch, error\n");
        for(OA oa : oa_list) {
            StringBuilder baseLine = new StringBuilder();
            baseLine.append(oa.name);
            baseLine.append(",");
            for(Layer l : (ArrayList<Layer>) oa.nn.getHiddenLayers()){
                baseLine.append(" ");
                baseLine.append(l.getNodeCount() - 1);
            }
            baseLine.append(", ");
            if (oa.algorithm instanceof SimulatedAnnealing){
                SimulatedAnnealing alg = (SimulatedAnnealing) oa.algorithm;
                baseLine.append(alg.getStartT());
                baseLine.append(", ");
                baseLine.append(alg.getCooling());
                baseLine.append(", 0, 0, 0, ");
            }
            else if (oa.algorithm instanceof StandardGeneticAlgorithm){
                baseLine.append("0 , 0, ");
                StandardGeneticAlgorithm alg = (StandardGeneticAlgorithm) oa.algorithm;
                baseLine.append(alg.getPopulationSize());
                baseLine.append(", ");
                baseLine.append(alg.getToMate());
                baseLine.append(", ");
                baseLine.append(alg.getToMutate());
                baseLine.append(", ");
            }
            else{
                baseLine.append("0, 0, 0, 0, 0, ");
            }
            String line_start = baseLine.toString();
            for (int i=0 ; i < oa.errors.length ; i++) {
                double e = oa.errors[i];
                sb.append(line_start);
                sb.append(i);
                sb.append(", ");
                sb.append(e);
                sb.append('\n');
            }
        }
        Utils.writeOutputToFile(outputDir, "StarcraftTestErrors" + outFileParticule + ".csv", sb.toString());
    }

    
    private static void train(OA oa) {
        oa.errors = new double[trainingIterations];
        for(int i = 0; i < trainingIterations; i++) {
            oa.errors[i] = oa.algorithm.train();

//            oa.errors[i] = 0;
//            for(Instance ins : instances) {
//                oa.nn.setInputValues(ins.getData());
//                oa.nn.run();
//
//                Instance output = ins.getLabel(), example = new Instance(oa.nn.getOutputValues());
//                System.out.println(oa.nn.getOutputValues().toString());
//                example.setLabel(new Instance(Double.parseDouble(oa.nn.getOutputValues().toString())));
//                oa.errors[i] += measure.value(output, example);
//            }
        }
    }

    private static Instance[] initializeInstances() {
        LinkedList<Instance> instances = new LinkedList<>();
        try {
            FileReader y_train_file = new FileReader("data/starcraft_y_train.csv");
            FileReader x_train_file = new FileReader("data/starcraft_x_train.csv");
            BufferedReader y_train_br = new BufferedReader(y_train_file);
            BufferedReader x_train_br = new BufferedReader(x_train_file);

            String y_train_line = y_train_br.readLine();
            String x_train_line = x_train_br.readLine();

            while (y_train_line != null && x_train_line != null){
                Scanner x_train_sc = new Scanner(x_train_line);
                x_train_sc.useDelimiter(",");
                double x_train_labels[] = new double[inputLayer];
                int i=0;
                while (x_train_sc.hasNext()){
                    x_train_labels[i] = Double.parseDouble(x_train_sc.next());
                    i++;
                }
                assert i == inputLayer;

                Scanner y_train_sc = new Scanner(x_train_line);
                y_train_sc.useDelimiter(",");
                double y_train_labels[] = new double[outputLayer];
                i=0;
                while (y_train_sc.hasNext()){
                    y_train_labels[i] = Double.parseDouble(y_train_sc.next());
                    i++;
                }
                assert i == outputLayer;

                instances.add(new Instance(new DenseVector(x_train_labels), new Instance(y_train_labels)));

                y_train_line = y_train_br.readLine();
                x_train_line = x_train_br.readLine();
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        // convert linked list to array
        Instance[] instances_array = new Instance[instances.size()];
        instances_array = instances.toArray(instances_array);

        return instances_array;
    }
}
