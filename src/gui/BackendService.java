package gui;

import core.algorithms.supervised.classification.DecisionTreeClassifier;
import core.algorithms.supervised.classification.RandomForestClassifier;
import core.algorithms.supervised.regression.DecisionTreeRegressor;
import core.algorithms.supervised.regression.ModularLinearRegression;
import core.algorithms.supervised.regression.RandomForestRegressor;
import core.algorithms.supervised.regression.costFunction.*;
import core.algorithms.supervised.regression.interfaces.CostFunction;
import core.algorithms.supervised.regression.interfaces.Optimizer;
import core.algorithms.supervised.regression.optimizer.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.stream.Collectors;

public class BackendService {

    // Custom JSON parsing logic to avoid external libraries and complex regex.
    private int parsePos = 0;

    private Map<String, Object> parseJsonObject(String json) {
        Map<String, Object> map = new HashMap<>();
        parsePos++; // Skip '{'
        while (parsePos < json.length() && json.charAt(parsePos) != '}') {
            String key = (String) parseValue(json);
            parsePos++; // Skip ':'
            Object value = parseValue(json);
            map.put(key, value);
            if (json.charAt(parsePos) == ',') {
                parsePos++; // Skip ','
            }
        }
        parsePos++; // Skip '}'
        return map;
    }

    private List<Object> parseJsonArray(String json) {
        List<Object> list = new ArrayList<>();
        parsePos++; // Skip '['
        while (parsePos < json.length() && json.charAt(parsePos) != ']') {
            list.add(parseValue(json));
            if (json.charAt(parsePos) == ',') {
                parsePos++; // Skip ','
            }
        }
        parsePos++; // Skip ']'
        return list;
    }

    private Object parseValue(String json) {
        // Trim whitespace
        while (parsePos < json.length() && Character.isWhitespace(json.charAt(parsePos))) {
            parsePos++;
        }

        char currentChar = json.charAt(parsePos);
        if (currentChar == '"') {
            parsePos++; // Skip '"'
            int end = json.indexOf('"', parsePos);
            String value = json.substring(parsePos, end);
            parsePos = end + 1;
            return value;
        } else if (currentChar == '{') {
            return parseJsonObject(json);
        } else if (currentChar == '[') {
            return parseJsonArray(json);
        } else {
            // It's a number or a literal
            int end = parsePos;
            while (end < json.length() && json.charAt(end) != ',' && json.charAt(end) != '}' && json.charAt(end) != ']') {
                end++;
            }
            String valueStr = json.substring(parsePos, end).trim();
            parsePos = end;
            if (valueStr.equals("true") || valueStr.equals("false")) {
                return Boolean.parseBoolean(valueStr);
            }
            return Double.parseDouble(valueStr);
        }
    }

    public String handleTrainRequest(String jsonRequest) {
        try {
            // 1. Parse the incoming JSON request
            this.parsePos = 0;
            Map<String, Object> request = (Map<String, Object>) parseValue(jsonRequest.trim());

            String algorithm = (String) request.get("algorithm");
            Map<String, Object> params = (Map<String, Object>) request.get("params");
            Map<String, Object> trainingData = (Map<String, Object>) request.get("trainingData");

            // 4. Extract training data
            List<List<Double>> xDataList = (List<List<Double>>) (Object) trainingData.get("X");
            List<Double> yDataList = (List<Double>) (Object) trainingData.get("y");
            double[][] X = xDataList.stream().map(l -> l.stream().mapToDouble(Double::doubleValue).toArray()).toArray(double[][]::new);
            double[] y_double = yDataList.stream().mapToDouble(Double::doubleValue).toArray();

            switch (algorithm) {
                case "linearRegression": {
                    Map<String, Object> costFunctionData = (Map<String, Object>) params.get("costFunction");
                    Map<String, Object> optimizerData = (Map<String, Object>) params.get("optimizer");

                    CostFunction costFunction = createCostFunction((String) costFunctionData.get("type"), (Map<String, Object>) costFunctionData.get("params"));
                    Optimizer optimizer = createOptimizer((String) optimizerData.get("type"), (Map<String, Object>) optimizerData.get("params"));

                    double learningRate = (Double) params.get("learningRate");
                    int epochs = ((Double) params.get("epochs")).intValue();
                    int numFeatures = ((Double) params.get("numFeatures")).intValue();

                    ModularLinearRegression model = new ModularLinearRegression(learningRate, epochs, numFeatures, costFunction, optimizer);
                    model.fit(X, y_double);

                    double[] weights = model.getWeights();
                    double bias = model.getBias();
                    double finalError = model.calculateError(X, y_double);

                    String weightsJson = Arrays.stream(weights)
                            .mapToObj(w -> String.format("%.4f", w))
                            .collect(Collectors.joining(", ", "[", "]"));

                    return """
                    {
                      "status": "success",
                      "message": "Model trained successfully!",
                      "learnedWeights": %s,
                      "learnedBias": %.4f,
                      "finalError": %.6f
                    }""".formatted(weightsJson, bias, finalError);
                }
                case "decisionTreeRegressor": {
                    int max_depth = ((Double) params.get("max_depth")).intValue();
                    int min_samples_split = ((Double) params.get("min_samples_split")).intValue();
                    DecisionTreeRegressor model = new DecisionTreeRegressor(max_depth, min_samples_split);
                    model.fit(X, y_double);
                    int depth = model.getDepth();
                    String treeJson = model.getTreeJson();
                    return """
                    {
                      "status": "success",
                      "message": "Decision Tree Regressor trained successfully!",
                      "treeDepth": %d,
                      "treeStructure": %s
                    }""".formatted(depth, treeJson);
                }
                case "decisionTreeClassifier": {
                    int[] y_int = Arrays.stream(y_double).mapToInt(d -> (int) d).toArray();
                    int max_depth = ((Double) params.get("max_depth")).intValue();
                    int min_samples_split = ((Double) params.get("min_samples_split")).intValue();
                    DecisionTreeClassifier model = new DecisionTreeClassifier(max_depth, min_samples_split);
                    model.fit(X, y_int);
                    int depth = model.getDepth();
                    String treeJson = model.getTreeJson();
                    return """
                    {
                      "status": "success",
                      "message": "Decision Tree Classifier trained successfully!",
                      "treeDepth": %d,
                      "treeStructure": %s
                    }""".formatted(depth, treeJson);
                }
                case "randomForestRegressor": {
                    int n_estimators = ((Double) params.get("n_estimators")).intValue();
                    int max_depth = ((Double) params.get("max_depth")).intValue();
                    int min_samples_split = ((Double) params.get("min_samples_split")).intValue();
                    RandomForestRegressor model = new RandomForestRegressor(n_estimators, max_depth, min_samples_split);
                    model.fit(X, y_double);
                    int numTrees = model.getNumberOfTrees();
                    String forestJson = model.getForestJson();
                    return """
                    {
                      "status": "success",
                      "message": "Random Forest Regressor trained successfully!",
                      "numberOfTrees": %d,
                      "forestStructure": %s
                    }""".formatted(numTrees, forestJson);
                }
                case "randomForestClassifier": {
                    int[] y_int = Arrays.stream(y_double).mapToInt(d -> (int) d).toArray();
                    int n_estimators = ((Double) params.get("n_estimators")).intValue();
                    int max_depth = ((Double) params.get("max_depth")).intValue();
                    int min_samples_split = ((Double) params.get("min_samples_split")).intValue();
                    RandomForestClassifier model = new RandomForestClassifier(n_estimators, max_depth, min_samples_split);
                    model.fit(X, y_int);
                    int numTrees = model.getNumberOfTrees();
                    String forestJson = model.getForestJson();
                    return """
                    {
                      "status": "success",
                      "message": "Random Forest Classifier trained successfully!",
                      "numberOfTrees": %d,
                      "forestStructure": %s
                    }""".formatted(numTrees, forestJson);
                }
                default:
                    throw new IllegalArgumentException("Unknown algorithm: " + algorithm);
            }

        } catch (Exception e) {
            e.printStackTrace();
            String errorMessage = e.getMessage().replace("\"", "'");
            return """
            {
              "status": "error",
              "message": "%s"
            }""".formatted(errorMessage);
        }
    }

    @SuppressWarnings("unchecked")
    public String handlePredictRequest(String jsonRequest) {
        try {
            this.parsePos = 0;
            Map<String, Object> request = (Map<String, Object>) parseValue(jsonRequest.trim());

            Map<String, Object> trainedModel = (Map<String, Object>) request.get("trainedModel");
            List<Double> predictionDataList = (List<Double>) (Object) request.get("predictionData");
            double[] predictionData = predictionDataList.stream().mapToDouble(Double::doubleValue).toArray();

            String algorithm = (String) trainedModel.get("algorithm");
            Map<String, Object> modelDetails = (Map<String, Object>) trainedModel.get("model_details");

            Object prediction;

            switch (algorithm) {
                case "linearRegression": {
                    List<Double> weightsList = (List<Double>) (Object) modelDetails.get("learnedWeights");
                    double[] weights = weightsList.stream().mapToDouble(Double::doubleValue).toArray();
                    double bias = (Double) modelDetails.get("learnedBias");
                    ModularLinearRegression model = new ModularLinearRegression(0, 0, 0, null, null);
                    model.setWeights(weights);
                    model.setBias(bias);
                    prediction = model.predictOne(predictionData);
                    break;
                }
                case "decisionTreeRegressor": {
                    Map<String, Object> treeStructure = (Map<String, Object>) modelDetails.get("treeStructure");
                    DecisionTreeRegressor model = DecisionTreeRegressor.fromJson(treeStructure);
                    prediction = model.predict_row(predictionData);
                    break;
                }
                case "decisionTreeClassifier": {
                    Map<String, Object> treeStructure = (Map<String, Object>) modelDetails.get("treeStructure");
                    DecisionTreeClassifier model = DecisionTreeClassifier.fromJson(treeStructure);
                    prediction = model.predict(predictionData);
                    break;
                }
                case "randomForestRegressor": {
                    List<Map<String, Object>> forestStructure = (List<Map<String, Object>>) (Object) modelDetails.get("forestStructure");
                    RandomForestRegressor model = RandomForestRegressor.fromJson(forestStructure);
                    prediction = model.predict_row(predictionData);
                    break;
                }
                case "randomForestClassifier": {
                    List<Map<String, Object>> forestStructure = (List<Map<String, Object>>) (Object) modelDetails.get("forestStructure");
                    RandomForestClassifier model = RandomForestClassifier.fromJson(forestStructure);
                    prediction = model.predict(predictionData);
                    break;
                }
                default:
                    throw new IllegalArgumentException("Unknown algorithm: " + algorithm);
            }

            return String.format("{\"status\": \"success\", \"prediction\": \"%s\"}", prediction.toString());

        } catch (Exception e) {
            e.printStackTrace();
            String errorMessage = e.getMessage().replace("\"", "'");
            return String.format("{\"status\": \"error\", \"message\": \"%s\"}", errorMessage);
        }
    }

    private CostFunction createCostFunction(String type, Map<String, Object> params) {
        if (params == null) params = new HashMap<>(); // Ensure params is not null
        switch (type) {
            case "MeanSquaredError": return new MeanSquaredError();
            case "MeanAbsoluteError": return new MeanAbsoluteError();
            case "HuberLoss":
                double delta = params.containsKey("delta") ? (Double) params.get("delta") : 1.0;
                return new HuberLoss(delta);
            case "LogCoshLoss": return new LogCoshLoss();
            default: throw new IllegalArgumentException("Unknown cost function: " + type);
        }
    }

    private Optimizer createOptimizer(String type, Map<String, Object> params) {
        if (params == null) params = new HashMap<>(); // Ensure params is not null
        switch (type) {
            case "BatchGradientDescent": return new BatchGradientDescent();
            case "MiniBatchGradientDescent":
                int batchSize = params.containsKey("batchSize") ? ((Double) params.get("batchSize")).intValue() : 32;
                return new MiniBatchGradientDescent(batchSize);
            case "MomentumOptimizer":
                double momentum = params.containsKey("momentum") ? (Double) params.get("momentum") : 0.9;
                return new MomentumOptimizer(momentum);
            case "AdamOptimizer":
                double beta1 = params.containsKey("beta1") ? (Double) params.get("beta1") : 0.9;
                double beta2 = params.containsKey("beta2") ? (Double) params.get("beta2") : 0.999;
                double epsilon = params.containsKey("epsilon") ? (Double) params.get("epsilon") : 1e-8;
                return new AdamOptimizer(beta1, beta2, epsilon);
            case "RMSPropOptimizer":
                double beta = params.containsKey("beta") ? (Double) params.get("beta") : 0.9;
                double rmsEpsilon = params.containsKey("epsilon") ? (Double) params.get("epsilon") : 1e-8;
                return new RMSPropOptimizer(beta, rmsEpsilon);
            case "AdagradOptimizer": return new AdagradOptimizer();
            default: throw new IllegalArgumentException("Unknown optimizer: " + type);
        }
    }
}
