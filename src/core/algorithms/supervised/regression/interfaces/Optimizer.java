package core.algorithms.supervised.regression.interfaces;

public interface Optimizer {
    void updateWeights(double[] weights, double[] gradients, double learningRate);
    void updateBias(double[] bias, double biasGradient, double learningRate);
    void reset(int numFeatures);
    String getName();
}