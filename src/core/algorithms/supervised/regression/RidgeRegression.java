package core.algorithms.supervised.regression;

public class RidgeRegression extends LinearRegression {
    private double lambda;
    //since we dont want it to be modified hence private

    public RidgeRegression(double learning_rate,int epochs,int n,double lambda){
        super(learning_rate,epochs,n);
        //super will call the constructor of the base class linear regression
        this.lambda=lambda;
    }

    @Override
    public void modify(double[] x,double y){
        double y_pred=predict(x);
        double e= y -y_pred;
        for(int i=0;i<weights.length;i++){
            //including the penalising term lambda 
            weights[i]+=learning_rate*(e*x[i]-2*lambda*weights[i]);

        }
        //modifying the bias 
        bias+=learning_rate*e;
    }
}
