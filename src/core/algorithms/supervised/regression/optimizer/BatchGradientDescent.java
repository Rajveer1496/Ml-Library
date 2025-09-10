package core.algorithms.supervised.regression.optimizer;

import core.algorithms.supervised.regression.interfaces.Optimizer;

public class BatchGradientDescent implements Optimizer {
    @Override
    public void updateWeights(double[] weights, double[] gradients, double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * gradients[i];
        }
    }

    @Override
    public void updateBias(double[] bias, double biasGradient, double learningRate) {
        bias[0] -= learningRate * biasGradient;
    }

    @Override
    public void reset(int numFeatures) {
        // No state to reset
    }

    @Override
    public String getName() {
        return "Batch Gradient Descent";
    }
}