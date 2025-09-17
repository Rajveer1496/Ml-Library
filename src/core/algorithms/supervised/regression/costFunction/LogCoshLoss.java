package core.algorithms.supervised.regression.costFunction;

import core.algorithms.supervised.regression.interfaces.CostFunction;

public class LogCoshLoss implements CostFunction {
    @Override
    public double calculateCost(double predicted, double actual) {
        double error = predicted - actual;
        return Math.log(Math.cosh(error));
    }

    @Override
    public double calculateGradient(double predicted, double actual, double feature) {
        double error = predicted - actual;
        return Math.tanh(error) * feature;
    }

    @Override
    public double calculateBiasGradient(double predicted, double actual) {
        double error = predicted - actual;
        return Math.tanh(error);
    }

    @Override
    public String getName() {
        return "Log-Cosh Loss";
    }
}