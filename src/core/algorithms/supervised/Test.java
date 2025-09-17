package core.algorithms.supervised;

import core.algorithms.supervised.regression.LinearRegression;
import core.algorithms.supervised.regression.DecisionTreeRegressor;
import core.algorithms.supervised.regression.RandomForestRegressor;
import core.algorithms.supervised.classification.DecisionTreeClassifier;
import core.algorithms.supervised.classification.RandomForestClassifier;

public class Test {
    public static void main(String[] args) {
        // -----------------------------
        // Regression tests
        // -----------------------------
        double[][] X_reg = {
            {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}
        };
        double[] y_reg = {
            2.1, 3.9, 6.2, 8.1, 10.3, 12.2, 13.9, 16.2, 17.8, 19.7
        };

        // 1. Linear Regression
        LinearRegression lr = new LinearRegression(0.01, 1000, 1);
        lr.fit(X_reg, y_reg);
        double[] preds_lr = lr.predict_all(new double[][]{{1.5}, {5.5}, {9.5}});
        System.out.println("Linear Regression Predictions:");
        for (double p : preds_lr) System.out.println(p);

        // 2. Decision Tree Regressor
        DecisionTreeRegressor dt = new DecisionTreeRegressor(5, 2);
        dt.fit(X_reg, y_reg);
        double[] preds_dt = dt.predict(new double[][]{{1.5}, {5.5}, {9.5}});
        System.out.println("\nDecision Tree Predictions:");
        for (double p : preds_dt) System.out.println(p);

        // 3. Random Forest Regressor
        RandomForestRegressor rf = new RandomForestRegressor(10, 5, 2);
        rf.fit(X_reg, y_reg);
        double[] preds_rf = rf.predict(new double[][]{{1.5}, {5.5}, {9.5}});
        System.out.println("\nRandom Forest Predictions:");
        for (double p : preds_rf) System.out.println(p);

        // -----------------------------
        // Classification tests
        // -----------------------------
        double[][] X_clf = {
            {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}
        };
        int[] y_clf = {
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1   // classify even=0, odd=1
        };

        // 4. Decision Tree Classifier
        DecisionTreeClassifier clf_dt = new DecisionTreeClassifier(5, 2);
        clf_dt.fit(X_clf, y_clf);
        int[] preds_clf_dt = clf_dt.predict(new double[][]{{11}, {12}, {13}});
        System.out.println("\nDecision Tree Classifier Predictions:");
        for (int p : preds_clf_dt) System.out.println(p);

        // 5. Random Forest Classifier
        RandomForestClassifier clf_rf = new RandomForestClassifier(10, 5, 2);
        clf_rf.fit(X_clf, y_clf);
        int[] preds_clf_rf = clf_rf.predict(new double[][]{{11}, {12}, {13}});
        System.out.println("\nRandom Forest Classifier Predictions:");
        for (int p : preds_clf_rf) System.out.println(p);
    }
}
