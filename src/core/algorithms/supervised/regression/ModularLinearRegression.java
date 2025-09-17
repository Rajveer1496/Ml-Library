package core.algorithms.supervised.regression;
import core.algorithms.supervised.regression.interfaces.CostFunction;
import core.algorithms.supervised.regression.interfaces.Optimizer;
import core.algorithms.supervised.regression.optimizer.MiniBatchGradientDescent;

import java.util.Random;

public class ModularLinearRegression {
    private double learningRate;
    private int epochs;
    private double[] weights;
    private double bias;
    private CostFunction costFunction;
    private Optimizer optimizer;
    private Random random;

    public ModularLinearRegression(double learningRate, int epochs, int numFeatures,
                                   CostFunction costFunction, Optimizer optimizer) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.weights = new double[numFeatures];
        this.bias = 0.0;
        this.costFunction = costFunction;
        this.optimizer = optimizer;
        this.random = new Random();

        // Initialize optimizer
        if (optimizer != null) {
            optimizer.reset(numFeatures);
        }
    }

    public double predictOne(double[] rowData) {
        double result = bias;
        for (int i = 0; i < weights.length; i++) {
            result += weights[i] * rowData[i];
        }
        return result;
    }

    public double[] predictAll(double[][] inputData) {
        double[] predictions = new double[inputData.length];
        for (int i = 0; i < inputData.length; i++) {
            predictions[i] = predictOne(inputData[i]);
        }
        return predictions;
    }

    public double calculateError(double[][] inputData, double[] outputData) {
        double totalCost = 0;
        for (int i = 0; i < inputData.length; i++) {
            double predicted = predictOne(inputData[i]);
            totalCost += costFunction.calculateCost(predicted, outputData[i]);
        }
        return totalCost / inputData.length;
    }

    public void fit(double[][] inputData, double[] outputData) {
        System.out.println("Training with " + costFunction.getName() + " and " + optimizer.getName());
        System.out.println("Initial error: " + calculateError(inputData, outputData));

        for (int epoch = 0; epoch < epochs; epoch++) {
            if (optimizer instanceof MiniBatchGradientDescent) {
                fitMiniBatch(inputData, outputData);
            } else {
                fitBatch(inputData, outputData);
            }

            // Print progress every 100 epochs
            if ((epoch + 1) % 100 == 0) {
                double error = calculateError(inputData, outputData);
                System.out.println("Epoch " + (epoch + 1) + ", Error: " + error);
            }
        }

        System.out.println("Final error: " + calculateError(inputData, outputData));
    }

    private void fitBatch(double[][] inputData, double[] outputData) {
        double[] weightGradients = new double[weights.length];
        double biasGradient = 0;

        // Calculate gradients for all samples
        for (int i = 0; i < inputData.length; i++) {
            double predicted = predictOne(inputData[i]);

            for (int j = 0; j < weights.length; j++) {
                weightGradients[j] += costFunction.calculateGradient(predicted, outputData[i], inputData[i][j]);
            }
            biasGradient += costFunction.calculateBiasGradient(predicted, outputData[i]);
        }

        // Average gradients
        for (int i = 0; i < weightGradients.length; i++) {
            weightGradients[i] /= inputData.length;
        }
        biasGradient /= inputData.length;

        // Update parameters
        optimizer.updateWeights(weights, weightGradients, learningRate);
        double[] biasArray = {bias};
        optimizer.updateBias(biasArray, biasGradient, learningRate);
        bias = biasArray[0];
    }

    private void fitMiniBatch(double[][] inputData, double[] outputData) {
        MiniBatchGradientDescent mbgd = (MiniBatchGradientDescent) optimizer;
        int batchSize = mbgd.getBatchSize();
        int numSamples = inputData.length;

        // Shuffle indices
        int[] indices = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }

        // Fisher-Yates shuffle
        for (int i = numSamples - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        // Process mini-batches
        for (int startIdx = 0; startIdx < numSamples; startIdx += batchSize) {
            int endIdx = Math.min(startIdx + batchSize, numSamples);
            int actualBatchSize = endIdx - startIdx;

            double[] weightGradients = new double[weights.length];
            double biasGradient = 0;

            // Calculate gradients for this mini-batch
            for (int i = startIdx; i < endIdx; i++) {
                int idx = indices[i];
                double predicted = predictOne(inputData[idx]);

                for (int j = 0; j < weights.length; j++) {
                    weightGradients[j] += costFunction.calculateGradient(predicted, outputData[idx], inputData[idx][j]);
                }
                biasGradient += costFunction.calculateBiasGradient(predicted, outputData[idx]);
            }

            // Update parameters
            optimizer.updateWeights(weights, weightGradients, learningRate);
            double[] biasArray = {bias};
            optimizer.updateBias(biasArray, biasGradient, learningRate);
            bias = biasArray[0];
        }
    }

    public double[] getWeights() {
        return weights.clone();
    }

    public double getBias() {
        return bias;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}