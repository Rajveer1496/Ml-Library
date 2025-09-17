package core.algorithms.supervised.regression;
import java.util.*;

public class RandomForestRegressor {
    private int numtrees;
    private int maxdepth;
    private int minsamplesplit;
    private List<DecisionTreeRegressor> trees;
    private Random rand;

    public RandomForestRegressor(int numtrees,int maxdepth,int minsamplesplit){
        this.numtrees=numtrees;
        this.maxdepth=maxdepth;
        this.minsamplesplit=minsamplesplit;
        this.trees=new ArrayList<>();
        this.rand=new Random();
    }

    public void fit ( double[][] X,double[] y){
        int numSamples=X.length;
        int numFeatures=X[0].length; // ADDED

        // ADDED: set maxFeatures for regression rule (numFeatures/3)
        int maxFeatures=Math.max(1,numFeatures/3);

        for(int i=0;i<numtrees;i++){
            double[][] Xsample=new double[numSamples][];
            double[] ysample=new double[numSamples];

            for(int j=0;j<numSamples;j++){
                int idx=rand.nextInt(numSamples);
                Xsample[j]=Arrays.copyOf(X[idx],X[idx].length); // safer copy
                ysample[j]=y[idx];
            }

            // ADDED: pass maxFeatures and rand to each tree
            DecisionTreeRegressor tree=new DecisionTreeRegressor(maxdepth,minsamplesplit,maxFeatures,rand);
            tree.fit(Xsample,ysample);
            trees.add(tree);
        }
    }

    public double predict_row(double[] x){
        double sum=0.0;
        for(DecisionTreeRegressor tree:trees){
            sum+=tree.predict_row(x);
        }
        return sum/trees.size();
    }

    public double[] predict(double[][] X){
        double[] predictions=new double[X.length];
        for(int i=0;i<X.length;i++){
            predictions[i]=predict_row(X[i]);
        }
        return predictions;
    }
}
