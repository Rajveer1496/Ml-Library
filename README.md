# ML Model Trainer GUI

This project provides a graphical user interface (GUI) for training and testing various machine learning models.

## How to Run the GUI

1.  **Compile the Java code:**

    ```bash
    javac -d out src/gui/*.java src/core/algorithms/supervised/regression/*.java src/core/algorithms/supervised/classification/*.java src/core/algorithms/supervised/regression/costFunction/*.java src/core/algorithms/supervised/regression/optimizer/*.java src/core/algorithms/supervised/regression/interfaces/*.java
    ```

2.  **Run the `GuiRunner` class:**

    ```bash
    java -cp out gui.GuiRunner
    ```

3.  **Open your web browser and navigate to:**

    [http://localhost:8080](http://localhost:8080)

## How to Use the GUI

1.  **Select an algorithm** from the dropdown menu.
2.  **Configure the parameters** for the selected algorithm.
3.  **Upload a CSV file** containing your training data. The last column of the CSV file should be the target variable.
4.  **Click the "Train Model" button** to train the model.
5.  **View the results** of the training in the "Results" section.
6.  **Make predictions** with the trained model by entering a comma-separated list of values in the "Make a Prediction" section and clicking the "Predict" button.

## Project Structure

*   `src/core/algorithms`: Contains the implementations of the machine learning algorithms.
*   `src/gui`: Contains the code for the GUI, including the HTML, CSS, and Java backend.
*   `src/gui/index.html`: The main HTML file for the GUI.
*   `src/gui/script.js`: The JavaScript file for the GUI.
*   `src/gui/GuiRunner.java`: The main class for running the GUI.
*   `src/gui/BackendService.java`: The class that handles the backend logic for training and prediction.

## Dependencies

This project uses only standard Java libraries, so there are no external dependencies to install.
