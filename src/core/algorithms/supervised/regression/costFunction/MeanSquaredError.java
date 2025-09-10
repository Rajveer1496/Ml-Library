package core.algorithms.supervised.regression.costFunction;

import core.algorithms.supervised.regression.interfaces.CostFunction;

public class MeanSquaredError implements CostFunction {
    @Override
    public double calculateCost(double predicted, double actual) {
        double error = predicted - actual;
        return error * error;
    }

    @Override
    public double calculateGradient(double predicted, double actual, double feature) {
        return 2 * (predicted - actual) * feature;
    }

    @Override
    public double calculateBiasGradient(double predicted, double actual) {
        return 2 * (predicted - actual);
    }

    @Override
    public String getName() {
        return "Mean Squared Error";
    }
}