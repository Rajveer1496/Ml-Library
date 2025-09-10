package core.algorithms.supervised.regression.costFunction;

import core.algorithms.supervised.regression.interfaces.CostFunction;

public class MeanAbsoluteError implements CostFunction {
    @Override
    public double calculateCost(double predicted, double actual) {
        return Math.abs(predicted - actual);
    }

    @Override
    public double calculateGradient(double predicted, double actual, double feature) {
        double error = predicted - actual;
        return (error > 0 ? 1 : -1) * feature;
    }

    @Override
    public double calculateBiasGradient(double predicted, double actual) {
        double error = predicted - actual;
        return (error > 0 ? 1 : -1);
    }

    @Override
    public String getName() {
        return "Mean Absolute Error";
    }
}