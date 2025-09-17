package core.algorithms.supervised.classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.HashMap;

public class RandomForestClassifier {
    private int numtrees;
    private int maxdepth;
    private int minsamplesplit;
    private List<DecisionTreeClassifier> trees;  //stores all decision trees
    private Random rand;   //used for bootstrap sampling


    public RandomForestClassifier(int numtrees,int maxdepth,int minsamplesplit){
        this.numtrees=numtrees;
        this.maxdepth=maxdepth;
        this.minsamplesplit=minsamplesplit;
        this.trees=new ArrayList<>();
        this.rand=new Random();
    }

    public void fit ( double[][] X,int[] y){
        //X -> 2D array of input features ( rows=samples,cols=features)
        //y -> labels (0 or 1)
        int numSamples=X.length;  //number of rows
        int numFeatures=X[0].length;   //no of cols (features) 

        for(int i=0;i<numtrees;i++){
            //bootstrap sampling
            //randomly sampled features
            double[][] Xsample=new double[numSamples][];
            int[] ysample=new int[numSamples];

            for(int j=0;j<numSamples;j++){
                int idx=rand.nextInt(numSamples);
                Xsample[j]=Arrays.copyOf(X[idx],numFeatures); // safer copy
                ysample[j]=y[idx];
            }

            //create n train tree
            DecisionTreeClassifier tree=new DecisionTreeClassifier(maxdepth,minsamplesplit);
            tree.fit(Xsample,ysample);
            trees.add(tree);
        }
    }
    //predict one row by majority vote
    public int predict(double[] x){
        Map<Integer,Integer> votes=new HashMap<>();
        for(DecisionTreeClassifier tree: trees){
            int pred=tree.predict(x);
            votes.put(pred,votes.getOrDefault(pred,0)+1);
        }

        int majorityclass=-1,maxvotes=0;
        for(int cls:votes.keySet()){
            int count=votes.get(cls);
            if(count>maxvotes){
                maxvotes=count;
                majorityclass=cls;
            }
        }
        return majorityclass;
    }

    //predict multiple rows
    public int[] predict(double[][] X){
        int[] predictions=new int[X.length];
        for(int i=0;i<X.length;i++){
            predictions[i]=predict(X[i]);
        }
        return predictions;
    }


}
