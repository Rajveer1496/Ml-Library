package core.algorithms.supervised.regression.optimizer;

import core.algorithms.supervised.regression.interfaces.Optimizer;

public class AdamOptimizer implements Optimizer {
    private double beta1, beta2, epsilon;
    private double[] m, v; // momentum and velocity for weights
    private double mBias, vBias; // momentum and velocity for bias
    private int t; // time step

    public AdamOptimizer(double beta1, double beta2, double epsilon) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0;
        this.mBias = 0.0;
        this.vBias = 0.0;
    }

    public AdamOptimizer() {
        this(0.9, 0.999, 1e-8); // default parameters
    }

    @Override
    public void updateWeights(double[] weights, double[] gradients, double learningRate) {
        if (m == null) {
            m = new double[weights.length];
            v = new double[weights.length];
        }

        t++;

        for (int i = 0; i < weights.length; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
            v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i];

            double mCorrected = m[i] / (1 - Math.pow(beta1, t));
            double vCorrected = v[i] / (1 - Math.pow(beta2, t));

            weights[i] -= learningRate * mCorrected / (Math.sqrt(vCorrected) + epsilon);
        }
    }

    @Override
    public void updateBias(double[] bias, double biasGradient, double learningRate) {
        mBias = beta1 * mBias + (1 - beta1) * biasGradient;
        vBias = beta2 * vBias + (1 - beta2) * biasGradient * biasGradient;

        double mCorrected = mBias / (1 - Math.pow(beta1, t));
        double vCorrected = vBias / (1 - Math.pow(beta2, t));

        bias[0] -= learningRate * mCorrected / (Math.sqrt(vCorrected) + epsilon);
    }

    @Override
    public void reset(int numFeatures) {
        m = new double[numFeatures];
        v = new double[numFeatures];
        mBias = 0.0;
        vBias = 0.0;
        t = 0;
    }

    @Override
    public String getName() {
        return "Adam Optimizer (β₁=" + beta1 + ", β₂=" + beta2 + ")";
    }
}