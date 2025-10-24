package core.algorithms.supervised.regression;

public class LassoRegression extends LinearRegression{
    private double lambda;

    public LassoRegression(double learning_rate,int epochs,int n,double lambda){
        super(learning_rate,epochs,n);
        this.lambda=lambda;
    }

    @Override
    public void modify(double[] x,double y){
        double y_pred=predict(x);
        double e=y-y_pred;
        for(int i=0;i<weights.length;i++){
            double sign=Math.signum(weights[i]);
            //we are only interested in the sign of the weights

            //sign decides whether to inc or dec the values of weights to make it tend to 0
            weights[i]+=learning_rate*(e*x[i]-lambda*sign);
        }  
        bias+=learning_rate*e;
    }
    
}
