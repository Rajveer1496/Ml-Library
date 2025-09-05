package core.algorithms.supervised.regression;

public class testLR {
    //THIS IS JUST FOR TESTING 
    public static void main(String[] args) {
        // dataset: one feature
        double[][] X = {
            {1}, {2}, {3}, {4}, {5}
        };
        double[] y = {5, 7, 9, 11, 13}; // y = 2x + 3

        // create model: learning rate, epochs, number of features
        LinearRegression model = new LinearRegression(0.01, 1000, 1);

        // train the model
        model.fit(X, y);

        // print learned weights and bias
        System.out.println("Learned weight: " + model.getWeights()[0]);
        System.out.println("Learned bias: " + model.getBias());

        // make a prediction
        double[] testPoint = {6};
        double prediction = model.predict_one_row(testPoint);
        System.out.println("Prediction for x=6: " + prediction);
    }
}
