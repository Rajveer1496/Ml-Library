package core.algorithms.supervised.regression.interfaces;

public interface CostFunction {
    double calculateCost(double predicted, double actual);
    double calculateGradient(double predicted, double actual, double feature);
    double calculateBiasGradient(double predicted, double actual);
    String getName();
}