package core.algorithms.supervised.regression.optimizer;

import core.algorithms.supervised.regression.interfaces.Optimizer;

public class AdagradOptimizer implements Optimizer {
    private double epsilon;
    private double[] G; // accumulated gradients for weights
    private double GBias; // accumulated gradients for bias

    public AdagradOptimizer(double epsilon) {
        this.epsilon = epsilon;
        this.GBias = 0.0;
    }

    public AdagradOptimizer() {
        this(1e-8); // default epsilon
    }

    @Override
    public void updateWeights(double[] weights, double[] gradients, double learningRate) {
        if (G == null) {
            G = new double[weights.length];
        }

        for (int i = 0; i < weights.length; i++) {
            G[i] += gradients[i] * gradients[i];
            weights[i] -= learningRate * gradients[i] / (Math.sqrt(G[i]) + epsilon);
        }
    }

    @Override
    public void updateBias(double[] bias, double biasGradient, double learningRate) {
        GBias += biasGradient * biasGradient;
        bias[0] -= learningRate * biasGradient / (Math.sqrt(GBias) + epsilon);
    }

    @Override
    public void reset(int numFeatures) {
        G = new double[numFeatures];
        GBias = 0.0;
    }

    @Override
    public String getName() {
        return "Adagrad Optimizer";
    }
}