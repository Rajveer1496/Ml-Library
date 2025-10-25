package core.algorithms.supervised.regression;

public class LinearRegression {
    protected double learning_rate;
    protected int epochs;
    protected double weights[];
    protected double bias;

    public LinearRegression(double learning_rate,int epochs,int n){
        this.learning_rate=learning_rate;
        this.epochs=epochs;
        this.weights=new double[n];
        // n-> no of features
        this.bias=0.0;
    }
    public LinearRegression() {
        this(0.01, 1000, 1);
    }

    //user does model.fit(int[][] dataset,int[] output)
    public double predict(double[] rowData){
        double ans=0.0;
        for(int i=0;i<weights.length;i++){
            ans+=weights[i]* rowData[i];
            //just like y=mx+c
        }
        ans=ans+bias;
        return ans;
    }

    public double[] predict_all(double[][] input_data){
        double[] prediction=new double[input_data.length]; //will contain the predictions made for each row 
        for(int i=0;i<input_data.length;i++){
            prediction[i]= predict(input_data[i]);
        }
        return prediction;
    }

    public double error_cal(double[][]input_data,double[] output_data){
        //error- mean square error
        double error=0;
        int n=output_data.length;
        for(int i=0;i<n;i++){
            //firstly we will predict the values for i/p data
            double predicted_ans= predict(input_data[i]);
            error+=(predicted_ans-output_data[i])*(predicted_ans-output_data[i]);
        }
        return error/n;

    }
    public void modify(double[] x,double y){
        //handling gradient descent
        double y_predicted= predict(x);
        double e=y-y_predicted;

        for(int i=0;i<weights.length;i++){
            weights[i]=weights[i]+(learning_rate*e*x[i]);
        }
        bias=bias+(learning_rate * e);
    }

    public void fit(double[][] input_data,double[] output_data){
        for(int i=0;i<epochs;i++){
            for(int j=0;j<input_data.length;j++){
                //traversing every row
                modify(input_data[j],output_data[j]);
            }
        }
    }

    public double[] getWeights() { 
        return weights; 
    }

    public double getBias() { 
        return bias;
    }

}
