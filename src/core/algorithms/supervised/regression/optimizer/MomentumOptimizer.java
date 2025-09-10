package core.algorithms.supervised.regression.optimizer;

import core.algorithms.supervised.regression.interfaces.Optimizer;

public class MomentumOptimizer implements Optimizer {
    private double momentum;
    private double[] velocityWeights;
    private double velocityBias;

    public MomentumOptimizer(double momentum) {
        this.momentum = momentum;
        this.velocityBias = 0.0;
    }

    public MomentumOptimizer() {
        this.momentum = 0.9; // default momentum
        this.velocityBias = 0.0;
    }

    @Override
    public void updateWeights(double[] weights, double[] gradients, double learningRate) {
        if (velocityWeights == null) {
            velocityWeights = new double[weights.length];
        }

        for (int i = 0; i < weights.length; i++) {
            velocityWeights[i] = momentum * velocityWeights[i] - learningRate * gradients[i];
            weights[i] += velocityWeights[i];
        }
    }

    @Override
    public void updateBias(double[] bias, double biasGradient, double learningRate) {
        velocityBias = momentum * velocityBias - learningRate * biasGradient;
        bias[0] += velocityBias;
    }

    @Override
    public void reset(int numFeatures) {
        velocityWeights = new double[numFeatures];
        velocityBias = 0.0;
    }

    @Override
    public String getName() {
        return "Momentum Optimizer (β=" + momentum + ")";
    }
}
