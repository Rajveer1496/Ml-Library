document.addEventListener('DOMContentLoaded', () => {
    const algorithmSelect = document.getElementById('algorithmSelect');
    const linearRegressionParams = document.getElementById('linearRegressionParams');
    const decisionTreeRegressorParams = document.getElementById('decisionTreeRegressorParams');
    const decisionTreeClassifierParams = document.getElementById('decisionTreeClassifierParams');
    const randomForestRegressorParams = document.getElementById('randomForestRegressorParams');
    const randomForestClassifierParams = document.getElementById('randomForestClassifierParams');
    const costFunctionSelect = document.getElementById('costFunctionSelect');
    const costFunctionParams = document.getElementById('costFunctionParams');
    const optimizerSelect = document.getElementById('optimizerSelect');
    const optimizerParams = document.getElementById('optimizerParams');
    const trainButton = document.getElementById('trainButton');
    const dataFile = document.getElementById('dataFile');
    const resultsDiv = document.getElementById('results');
    const predictionSection = document.getElementById('predictionSection');
    const predictButton = document.getElementById('predictButton');
    const predictionData = document.getElementById('predictionData');
    const predictionResult = document.getElementById('predictionResult');

    let trainedModel = null;

    // Function to show/hide parameter sections based on algorithm selection
    algorithmSelect.addEventListener('change', () => {
        linearRegressionParams.style.display = 'none';
        decisionTreeRegressorParams.style.display = 'none';
        decisionTreeClassifierParams.style.display = 'none';
        randomForestRegressorParams.style.display = 'none';
        randomForestClassifierParams.style.display = 'none';

        const selectedAlgorithm = algorithmSelect.value;
        if (selectedAlgorithm === 'linearRegression') {
            linearRegressionParams.style.display = 'block';
        } else if (selectedAlgorithm === 'decisionTreeRegressor') {
            decisionTreeRegressorParams.style.display = 'block';
        } else if (selectedAlgorithm === 'decisionTreeClassifier') {
            decisionTreeClassifierParams.style.display = 'block';
        } else if (selectedAlgorithm === 'randomForestRegressor') {
            randomForestRegressorParams.style.display = 'block';
        } else if (selectedAlgorithm === 'randomForestClassifier') {
            randomForestClassifierParams.style.display = 'block';
        }
    });

    // Function to dynamically generate cost function specific parameters
    costFunctionSelect.addEventListener('change', () => {
        const selectedCostFunction = costFunctionSelect.value;
        costFunctionParams.innerHTML = '<h3>Cost Function Specific Parameters</h3>'; // Clear previous params
        costFunctionParams.style.display = 'none';

        if (selectedCostFunction === 'HuberLoss') {
            costFunctionParams.style.display = 'block';
            const deltaInput = document.createElement('div');
            deltaInput.classList.add('form-group');
            deltaInput.innerHTML = `
                <label for="huberDelta">Delta:</label>
                <input type="number" id="huberDelta" value="1.0" step="0.1">
            `;
            costFunctionParams.appendChild(deltaInput);
        }
    });

    // Function to dynamically generate optimizer specific parameters
    optimizerSelect.addEventListener('change', () => {
        const selectedOptimizer = optimizerSelect.value;
        optimizerParams.innerHTML = '<h3>Optimizer Specific Parameters</h3>'; // Clear previous params
        optimizerParams.style.display = 'none';

        if (selectedOptimizer === 'MiniBatchGradientDescent') {
            optimizerParams.style.display = 'block';
            const batchSizeInput = document.createElement('div');
            batchSizeInput.classList.add('form-group');
            batchSizeInput.innerHTML = `
                <label for="batchSize">Batch Size:</label>
                <input type="number" id="batchSize" value="32" step="1">
            `;
            optimizerParams.appendChild(batchSizeInput);
        } else if (selectedOptimizer === 'MomentumOptimizer') {
            optimizerParams.style.display = 'block';
            const momentumInput = document.createElement('div');
            momentumInput.classList.add('form-group');
            momentumInput.innerHTML = `
                <label for="momentum">Momentum (beta):</label>
                <input type="number" id="momentum" value="0.9" step="0.01">
            `;
            optimizerParams.appendChild(momentumInput);
        } else if (selectedOptimizer === 'AdamOptimizer') {
            optimizerParams.style.display = 'block';
            const beta1Input = document.createElement('div');
            beta1Input.classList.add('form-group');
            beta1Input.innerHTML = `
                <label for="adamBeta1">Beta1:</label>
                <input type="number" id="adamBeta1" value="0.9" step="0.001">
            `;
            optimizerParams.appendChild(beta1Input);

            const beta2Input = document.createElement('div');
            beta2Input.classList.add('form-group');
            beta2Input.innerHTML = `
                <label for="adamBeta2">Beta2:</label>
                <input type="number" id="adamBeta2" value="0.999" step="0.001">
            `;
            optimizerParams.appendChild(beta2Input);

            const epsilonInput = document.createElement('div');
            epsilonInput.classList.add('form-group');
            epsilonInput.innerHTML = `
                <label for="adamEpsilon">Epsilon:</label>
                <input type="number" id="adamEpsilon" value="1e-8" step="1e-9">
            `;
            optimizerParams.appendChild(epsilonInput);

        } else if (selectedOptimizer === 'RMSPropOptimizer') {
            optimizerParams.style.display = 'block';
            const betaInput = document.createElement('div');
            betaInput.classList.add('form-group');
            betaInput.innerHTML = `
                <label for="rmsPropBeta">Beta:</label>
                <input type="number" id="rmsPropBeta" value="0.9" step="0.01">
            `;
            optimizerParams.appendChild(betaInput);

            const epsilonInput = document.createElement('div');
            epsilonInput.classList.add('form-group');
            epsilonInput.innerHTML = `
                <label for="rmsPropEpsilon">Epsilon:</label>
                <input type="number" id="rmsPropEpsilon" value="1e-8" step="1e-9">
            `;
            optimizerParams.appendChild(epsilonInput);
        }
    });

    trainButton.addEventListener('click', async () => {
        resultsDiv.textContent = 'Training model...';
        predictionSection.style.display = 'none';
        trainedModel = null;

        const selectedAlgorithm = algorithmSelect.value;
        if (!selectedAlgorithm) {
            resultsDiv.textContent = 'Please select an algorithm.';
            return;
        }

        const file = dataFile.files[0];
        if (!file) {
            resultsDiv.textContent = 'Please upload a data file.';
            return;
        }

        let params = {};
        if (selectedAlgorithm === 'linearRegression') {
            const selectedCostFunction = costFunctionSelect.value;
            const selectedOptimizer = optimizerSelect.value;
            if (!selectedCostFunction || !selectedOptimizer) {
                resultsDiv.textContent = 'Please select a cost function and an optimizer for Linear Regression.';
                return;
            }

            let costFunctionParamsData = {};
            if (selectedCostFunction === 'HuberLoss') {
                costFunctionParamsData.delta = parseFloat(document.getElementById('huberDelta').value);
            }

            let optimizerParamsData = {};
            if (selectedOptimizer === 'MiniBatchGradientDescent') {
                optimizerParamsData.batchSize = parseInt(document.getElementById('batchSize').value);
            } else if (selectedOptimizer === 'MomentumOptimizer') {
                optimizerParamsData.momentum = parseFloat(document.getElementById('momentum').value);
            } else if (selectedOptimizer === 'AdamOptimizer') {
                optimizerParamsData.beta1 = parseFloat(document.getElementById('adamBeta1').value);
                optimizerParamsData.beta2 = parseFloat(document.getElementById('adamBeta2').value);
                optimizerParamsData.epsilon = parseFloat(document.getElementById('adamEpsilon').value);
            } else if (selectedOptimizer === 'RMSPropOptimizer') {
                optimizerParamsData.beta = parseFloat(document.getElementById('rmsPropBeta').value);
                optimizerParamsData.epsilon = parseFloat(document.getElementById('rmsPropEpsilon').value);
            }

            params = {
                learningRate: parseFloat(document.getElementById('learningRate').value),
                epochs: parseInt(document.getElementById('epochs').value),
                costFunction: {
                    type: selectedCostFunction,
                    params: costFunctionParamsData
                },
                optimizer: {
                    type: selectedOptimizer,
                    params: optimizerParamsData
                }
            };
        } else if (selectedAlgorithm === 'decisionTreeRegressor') {
            params = {
                max_depth: parseInt(document.getElementById('dt_reg_max_depth').value),
                min_samples_split: parseInt(document.getElementById('dt_reg_min_samples_split').value)
            };
        } else if (selectedAlgorithm === 'decisionTreeClassifier') {
            params = {
                max_depth: parseInt(document.getElementById('dt_clf_max_depth').value),
                min_samples_split: parseInt(document.getElementById('dt_clf_min_samples_split').value)
            };
        } else if (selectedAlgorithm === 'randomForestRegressor') {
            params = {
                n_estimators: parseInt(document.getElementById('rf_reg_n_estimators').value),
                max_depth: parseInt(document.getElementById('rf_reg_max_depth').value),
                min_samples_split: parseInt(document.getElementById('rf_reg_min_samples_split').value)
            };
        } else if (selectedAlgorithm === 'randomForestClassifier') {
            params = {
                n_estimators: parseInt(document.getElementById('rf_clf_n_estimators').value),
                max_depth: parseInt(document.getElementById('rf_clf_max_depth').value),
                min_samples_split: parseInt(document.getElementById('rf_clf_min_samples_split').value)
            };
        }

        const reader = new FileReader();
        reader.onload = async (e) => {
            const csvContent = e.target.result;
            const lines = csvContent.trim().split('\n');
            const data = lines.map(line => line.split(',').map(Number));

            const X = data.map(row => row.slice(0, row.length - 1));
            const y = data.map(row => row[row.length - 1]);
            const numFeatures = X[0].length;
            params.numFeatures = numFeatures;

            const requestBody = {
                algorithm: selectedAlgorithm,
                params: params,
                trainingData: {
                    X: X,
                    y: y
                }
            };

            try {
                const response = await fetch('/train', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                resultsDiv.textContent = JSON.stringify(result, null, 2);

                if (result.status === 'success') {
                    trainedModel = {
                        algorithm: selectedAlgorithm,
                        params: params,
                        model_details: result
                    };
                    predictionSection.style.display = 'block';
                }

            } catch (error) {
                console.error('Error during training:', error);
                resultsDiv.textContent = `Error: ${error.message}. Make sure your backend server is running and accessible.`;
            }
        };
        reader.readAsText(file);
    });

    predictButton.addEventListener('click', async () => {
        if (!trainedModel) {
            predictionResult.textContent = 'Train a model first before making a prediction.';
            return;
        }

        const dataToPredict = predictionData.value.split(',').map(Number);
        if (dataToPredict.some(isNaN)) {
            predictionResult.textContent = 'Invalid input. Please enter comma-separated numbers.';
            return;
        }

        const requestBody = {
            trainedModel: trainedModel,
            predictionData: dataToPredict
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            predictionResult.textContent = JSON.stringify(result, null, 2);

        } catch (error) {
            console.error('Error during prediction:', error);
            predictionResult.textContent = `Error: ${error.message}`;
        }
    });
});
