package core.algorithms.supervised.regression;

public class ElasticNetRegression extends LinearRegression {
    private double lambda1;
    private double lambda2;

    public ElasticNetRegression(double learning_rate,int epochs,int n,double lambda1,double lambda2){
        super(learning_rate, epochs, n);
        this.lambda1=lambda1;
        this.lambda2=lambda2;
    }

    @Override
    public void modify(double[] x,double y){
        double y_pred=predict(x);
        double e=y-y_pred;
        for(int i=0;i<weights.length;i++){
            double sign=Math.signum(weights[i]);
            weights[i]+=learning_rate*(e*x[i]-lambda1*sign - 2*lambda2*weights[i]);
            // linear regression  weights[i]+=learning_rate*e*x[i]
            //ridge regression weights[i]+=learning_rate*(e*x[i]- 2*lambda2*weights[i]);
            //lasso regression   weights[i]+=learning_rate*(e*x[i]-lambda1*sign )
            //combining ridge n lasso we get elastic net 
        }
        bias+=learning_rate*e;
    }


    
}
