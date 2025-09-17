package core.algorithms.supervised.regression.costFunction;

import core.algorithms.supervised.regression.interfaces.CostFunction;

public class HuberLoss implements CostFunction {
    private double delta;

    public HuberLoss(double delta) {
        this.delta = delta;
    }

    public HuberLoss() {
        this.delta = 1.0; // default delta
    }

    @Override
    public double calculateCost(double predicted, double actual) {
        double error = Math.abs(predicted - actual);
        if (error <= delta) {
            return 0.5 * error * error;
        } else {
            return delta * error - 0.5 * delta * delta;
        }
    }

    @Override
    public double calculateGradient(double predicted, double actual, double feature) {
        double error = predicted - actual;
        if (Math.abs(error) <= delta) {
            return error * feature;
        } else {
            return delta * (error > 0 ? 1 : -1) * feature;
        }
    }

    @Override
    public double calculateBiasGradient(double predicted, double actual) {
        double error = predicted - actual;
        if (Math.abs(error) <= delta) {
            return error;
        } else {
            return delta * (error > 0 ? 1 : -1);
        }
    }

    @Override
    public String getName() {
        return "Huber Loss (δ=" + delta + ")";
    }
}
