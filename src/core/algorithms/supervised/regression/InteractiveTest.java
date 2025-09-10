package core.algorithms.supervised.regression;
import core.algorithms.supervised.regression.costFunction.HuberLoss;
import core.algorithms.supervised.regression.costFunction.LogCoshLoss;
import core.algorithms.supervised.regression.costFunction.MeanAbsoluteError;
import core.algorithms.supervised.regression.costFunction.MeanSquaredError;
import core.algorithms.supervised.regression.interfaces.CostFunction;
import core.algorithms.supervised.regression.interfaces.Optimizer;
import core.algorithms.supervised.regression.optimizer.*;

import java.util.Scanner;

public class InteractiveTest {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Sample dataset: y = 2x + 3
        double[][] X = {
                {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}
        };
        double[] y = {5, 7, 9, 11, 13, 15, 17, 19, 21, 23};

        System.out.println("=== Modular Linear Regression ===");
        System.out.println("Dataset: y = 2x + 3");
        System.out.println();

        // Select Cost Function
        System.out.println("Available Cost Functions:");
        System.out.println("1. Mean Squared Error (MSE)");
        System.out.println("2. Mean Absolute Error (MAE)");
//        System.out.println("3. Huber Loss");
//        System.out.println("4. Log-Cosh Loss");
        System.out.print("Choose cost function (1-2): ");

        int costChoice = scanner.nextInt();
        CostFunction costFunction = createCostFunction(costChoice, scanner);

        // Select Optimizer
        System.out.println("\nAvailable Optimizers:");
        System.out.println("1. Batch Gradient Descent");
        System.out.println("2. Mini-Batch Gradient Descent");
        System.out.println("3. Momentum Optimizer");
//        System.out.println("4. Adam Optimizer");
//        System.out.println("5. RMSProp Optimizer");
//        System.out.println("6. Adagrad Optimizer");
        System.out.print("Choose optimizer (1-3): ");

        int optimizerChoice = scanner.nextInt();
        Optimizer optimizer = createOptimizer(optimizerChoice, scanner);

        // Get training parameters
        System.out.print("\nEnter learning rate (e.g., 0.01): ");
        double learningRate = scanner.nextDouble();

        System.out.print("Enter number of epochs (e.g., 1000): ");
        int epochs = scanner.nextInt();

        // Create and train model
        ModularLinearRegression model = new ModularLinearRegression(
                learningRate, epochs, 1, costFunction, optimizer);

        System.out.println("\nTraining model...");
        model.fit(X, y);

        // Display results
        System.out.println("\n=== Results ===");
        System.out.printf("Learned weight: %.4f (expected: 2.0000)\n", model.getWeights()[0]);
        System.out.printf("Learned bias: %.4f (expected: 3.0000)\n", model.getBias());

        // Make predictions
        System.out.println("\n=== Predictions ===");
        double[] testPoints = {11, 12, 15};
        for (double testPoint : testPoints) {
            double[] testX = {testPoint};
            double prediction = model.predictOne(testX);
            double expected = 2 * testPoint + 3;
            System.out.printf("x=%.0f: predicted=%.4f, expected=%.4f\n",
                    testPoint, prediction, expected);
        }

        scanner.close();
    }

    private static CostFunction createCostFunction(int choice, Scanner scanner) {
        switch (choice) {
            case 1:
                return new MeanSquaredError();
            case 2:
                return new MeanAbsoluteError();
            case 3:
                System.out.print("Enter delta for Huber Loss (e.g., 1.0): ");
                double delta = scanner.nextDouble();
                return new HuberLoss(delta);
            case 4:
                return new LogCoshLoss();
            default:
                System.out.println("Invalid choice. Using MSE.");
                return new MeanSquaredError();
        }
    }

    private static Optimizer createOptimizer(int choice, Scanner scanner) {
        switch (choice) {
            case 1:
                return new BatchGradientDescent();
            case 2:
                System.out.print("Enter batch size (e.g., 32): ");
                int batchSize = scanner.nextInt();
                return new MiniBatchGradientDescent(batchSize);
            case 3:
                System.out.print("Enter momentum (e.g., 0.9): ");
                double momentum = scanner.nextDouble();
                return new MomentumOptimizer(momentum);
            case 4:
                System.out.print("Enter beta1 (e.g., 0.9): ");
                double beta1 = scanner.nextDouble();
                System.out.print("Enter beta2 (e.g., 0.999): ");
                double beta2 = scanner.nextDouble();
                return new AdamOptimizer(beta1, beta2, 1e-8);
            case 5:
                System.out.print("Enter beta for RMSProp (e.g., 0.9): ");
                double beta = scanner.nextDouble();
                return new RMSPropOptimizer(beta, 1e-8);
            case 6:
                return new AdagradOptimizer();
            default:
                System.out.println("Invalid choice. Using Batch GD.");
                return new BatchGradientDescent();
        }
    }
}