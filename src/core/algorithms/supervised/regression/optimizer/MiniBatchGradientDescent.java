package core.algorithms.supervised.regression.optimizer;

import core.algorithms.supervised.regression.interfaces.Optimizer;

public class MiniBatchGradientDescent implements Optimizer {
    private int batchSize;

    public MiniBatchGradientDescent(int batchSize) {
        this.batchSize = batchSize;
    }

    public MiniBatchGradientDescent() {
        this.batchSize = 32; // default batch size
    }

    @Override
    public void updateWeights(double[] weights, double[] gradients, double learningRate) {
        // Average gradients over batch
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * gradients[i] / batchSize;
        }
    }

    @Override
    public void updateBias(double[] bias, double biasGradient, double learningRate) {
        bias[0] -= learningRate * biasGradient / batchSize;
    }

    @Override
    public void reset(int numFeatures) {
        // No state to reset
    }

    @Override
    public String getName() {
        return "Mini-Batch Gradient Descent (batch size=" + batchSize + ")";
    }

    public int getBatchSize() {
        return batchSize;
    }
}