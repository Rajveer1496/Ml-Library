package core.algorithms.supervised.regression.optimizer;

import core.algorithms.supervised.regression.interfaces.Optimizer;

public class RMSPropOptimizer implements Optimizer {
    private double beta, epsilon;
    private double[] v; // velocity for weights
    private double vBias; // velocity for bias

    public RMSPropOptimizer(double beta, double epsilon) {
        this.beta = beta;
        this.epsilon = epsilon;
        this.vBias = 0.0;
    }

    public RMSPropOptimizer() {
        this(0.9, 1e-8); // default parameters
    }

    @Override
    public void updateWeights(double[] weights, double[] gradients, double learningRate) {
        if (v == null) {
            v = new double[weights.length];
        }

        for (int i = 0; i < weights.length; i++) {
            v[i] = beta * v[i] + (1 - beta) * gradients[i] * gradients[i];
            weights[i] -= learningRate * gradients[i] / (Math.sqrt(v[i]) + epsilon);
        }
    }

    @Override
    public void updateBias(double[] bias, double biasGradient, double learningRate) {
        vBias = beta * vBias + (1 - beta) * biasGradient * biasGradient;
        bias[0] -= learningRate * biasGradient / (Math.sqrt(vBias) + epsilon);
    }

    @Override
    public void reset(int numFeatures) {
        v = new double[numFeatures];
        vBias = 0.0;
    }

    @Override
    public String getName() {
        return "RMSProp Optimizer (β=" + beta + ")";
    }
}