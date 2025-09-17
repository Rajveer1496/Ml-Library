package core.algorithms.supervised.regression;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

public class DecisionTreeRegressor {
    private int maxdepth;
    private int minsamplesplit;
    private Node root;

    private int maxFeatures;   // how many features to consider at each split
    private Random rand;       // random generator for feature sampling

    public DecisionTreeRegressor(int maxdepth,int minsamplesplit){
        this(maxdepth, minsamplesplit, -1, new Random());
    }

    public DecisionTreeRegressor(int maxdepth,int minsamplesplit,int maxFeatures,Random rand){
        this.maxdepth=maxdepth;
        this.minsamplesplit=minsamplesplit;
        this.maxFeatures=maxFeatures; // -1 means all features
        this.rand=rand;
    }

    private DecisionTreeRegressor(Node root) {
        this.root = root;
    }

    private static class Node{
        int featureidx;
        double threshold;
        Node left;
        Node right;
        double value;
        boolean isleaf;

        Node(double value){
            this.value=value;
            this.isleaf=true;
        }

        Node(int featureidx,double threshold,Node left,Node right){
            this.featureidx=featureidx;
            this.threshold=threshold;
            this.left=left;
            this.right=right;
            this.isleaf=false;
        }
    }

    double mse(double[] y){
        double mean=0,sum=0,error=0;
        for(double val:y){
            sum+=val;
        }
        mean=sum/y.length;
        sum=0;
        for(double val:y){
            sum+=(val-mean)*(val-mean);
        }
        return sum/y.length;
    }

    public void fit(double[][] x,double[] y){
        root=buildtree(x,y,0);
    }

    private Node buildtree(double[][] x,double[] y,int depth){
        if(depth>=maxdepth || y.length<minsamplesplit){
            double sum=0;
            for(double val:y){
                sum+=val;
            }
            return new Node(sum/y.length);
        }

        SplitResult bestsplit=findbestsplit(x,y);
        if(bestsplit==null){
            double sum=0;
            for(double val:y){
                sum+=val;
            }
            return new Node(sum/y.length);
        }

        Node left=buildtree(bestsplit.Xleft,bestsplit.yleft, depth+1);
        Node right=buildtree(bestsplit.Xright,bestsplit.yright, depth+1);

        return new Node(bestsplit.featureidx,bestsplit.threshold,left,right);
    }

    public double predict_row(double[] x){
        return traverse(root,x);
    }

    public double traverse(Node node,double[] x){
        if(node.isleaf){
            return node.value;
        }
        if(x[node.featureidx]<node.threshold){
            return traverse(node.left,x);
        }
        else{
            return traverse(node.right, x);
        }
    }

    public double[] predict(double[][] x){
        double[] predictions=new double[x.length];
        for(int i=0;i<x.length;i++){
            predictions[i]=predict_row(x[i]);
        }
        return predictions;
    }

    private class SplitResult{
        int featureidx;
        double threshold;
        double[][] Xleft,Xright;
        double[] yleft,yright;

        SplitResult(int featureidx,double threshold,double[][] Xleft,double[] yleft,double[][] Xright,double[] yright){
            this.featureidx=featureidx;
            this.threshold=threshold;
            this.Xleft=Xleft;
            this.yleft=yleft;
            this.Xright=Xright;
            this.yright=yright;
        }
    }

    private SplitResult findbestsplit(double[][] X,double[] y){
        int numSamples=X.length;
        int numFeatures=X[0].length;

        //pick random subset of features
        int featuresToTry=(maxFeatures==-1)?numFeatures:Math.min(maxFeatures,numFeatures);
        List<Integer> featureIndices=new ArrayList<>();
        while(featureIndices.size()<featuresToTry){
            int f=rand.nextInt(numFeatures);
            if(!featureIndices.contains(f)){
                featureIndices.add(f);
            }
        }

        double bestmse=Double.MAX_VALUE; 
        SplitResult bestsplit=null;

        // loop only over sampled features
        for(int feature:featureIndices){
            double[] featurevalues=new double[numSamples];
            for(int i=0;i<numSamples;i++){
                featurevalues[i]=X[i][feature];
            }
            Arrays.sort(featurevalues);

            //use midpoints between values
            for(int t=1;t<featurevalues.length;t++){
                double threshold=(featurevalues[t-1]+featurevalues[t])/2.0;

                List<double[]> leftXlist=new ArrayList<>();
                List<Double> leftYlist=new ArrayList<>();
                List<double[]> rightXlist=new ArrayList<>();
                List<Double> rightYlist=new ArrayList<>();
                
                for(int i=0;i<numSamples;i++){
                    if(X[i][feature]<threshold){
                        leftXlist.add(X[i]);
                        leftYlist.add(y[i]);
                    }
                    else{
                        rightXlist.add(X[i]);
                        rightYlist.add(y[i]);
                    }
                }

                if(leftYlist.size()==0 || rightYlist.size()==0){
                    continue;
                }

                double[] leftY=new double[leftYlist.size()];
                for(int i=0;i<leftYlist.size();i++){
                    leftY[i]=leftYlist.get(i);
                }

                double[] rightY=new double[rightYlist.size()];
                for(int i=0;i<rightYlist.size();i++){
                    rightY[i]=rightYlist.get(i);
                }

                double mseleft=mse(leftY);
                double mseright=mse(rightY);

                double weightedmse=((double) leftY.length/numSamples) * mseleft + ((double) rightY.length / numSamples)* mseright;

                if(weightedmse<bestmse){
                    bestmse=weightedmse;
                    double[][] Xleft=leftXlist.toArray(new double[0][]);
                    double[][] Xright=rightXlist.toArray(new double[0][]);

                    bestsplit=new SplitResult(feature,threshold, Xleft, leftY, Xright, rightY);
                }
            }
        }
        return bestsplit;
    }

    public int getDepth() {
        return getDepth(root);
    }

    private int getDepth(Node node) {
        if (node == null) {
            return 0;
        }
        if (node.isleaf) {
            return 1;
        }
        return 1 + Math.max(getDepth(node.left), getDepth(node.right));
    }

    public String getTreeJson() {
        return nodeToJson(root);
    }

    private String nodeToJson(Node node) {
        if (node.isleaf) {
            return String.format("{\"type\": \"leaf\", \"value\": %.4f}", node.value);
        } else {
            String leftJson = nodeToJson(node.left);
            String rightJson = nodeToJson(node.right);
            return String.format("{\"type\": \"split\", \"feature_index\": %d, \"threshold\": %.4f, \"left\": %s, \"right\": %s}",
                    node.featureidx, node.threshold, leftJson, rightJson);
        }
    }

    public static DecisionTreeRegressor fromJson(Map<String, Object> treeJson) {
        Node root = nodeFromJson(treeJson);
        return new DecisionTreeRegressor(root);
    }

    @SuppressWarnings("unchecked")
    private static Node nodeFromJson(Map<String, Object> nodeJson) {
        String type = (String) nodeJson.get("type");
        if ("leaf".equals(type)) {
            return new Node((Double) nodeJson.get("value"));
        } else {
            int featureIndex = ((Double) nodeJson.get("feature_index")).intValue();
            double threshold = (Double) nodeJson.get("threshold");
            Node left = nodeFromJson((Map<String, Object>) nodeJson.get("left"));
            Node right = nodeFromJson((Map<String, Object>) nodeJson.get("right"));
            return new Node(featureIndex, threshold, left, right);
        }
    }
}