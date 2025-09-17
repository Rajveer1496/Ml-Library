package core.algorithms.supervised.classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DecisionTreeClassifier {

    private int maxdepth;  //max tree depth 
    private int minsamplesplit; //min samples needed for a split
    private Node root;

    public DecisionTreeClassifier(int maxdepth, int minsamplesplit) {
        this.maxdepth = maxdepth;
        this.minsamplesplit = minsamplesplit;
    }

    private DecisionTreeClassifier(Node root) {
        this.root = root;
    }

    // node
    private static class Node{
        boolean isleaf;
        int fidx; //feature index : idx chosen for fitting
        double threshold;   //cutoff value for feature
        Node left,right;
        int classlabel;    // majority vote (only set for leaf nodes)

        //constructor for leaf node
        Node(int classlabel){
            this.isleaf=true;
            this.classlabel=classlabel;
        }

        //for decision nodes
        Node(int fidx,double threshold,Node left,Node right){
            this.isleaf=false;
            this.fidx=fidx;
            this.threshold=threshold;
            this.left=left;
            this.right=right;
        }
    }

    //train
    public void fit(double[][] X,int[] y){
        this.root=buildtree(X,y,0);
        //initial depth=0
    }

    //build tree recursively
    private Node buildtree(double[][] X,int[] y,int depth){
        if(depth>=maxdepth || y.length<minsamplesplit || ispure(y)){
            return new Node(majorityclass(y));
            //make leaf node with majorityclass (votes)
        }

        SplitResult split=findbestsplit(X,y);
        if(split==null){
            return new Node(majorityclass(y));
            //if no good split
        }

        //else split the dataset 
        Node left=buildtree(split.Xleft,split.yleft, depth+1);
        Node right=buildtree(split.Xright,split.yright, depth+1);
        //decision node
        return new Node(split.fidx,split.threshold,left,right);
    }

    //if all labels are same
    private boolean ispure(int[] y){
        int first=y[0]; 
        for(int val:y){
            if(val!=first){
                return false;
            }
        }
        return true;
    }

    //counts clasees and returns majority class
    private int majorityclass(int[] y){
        Map<Integer,Integer> count=new HashMap<>();
        //store keyclass label(0,1,2...) and value=frequency
        for(int val: y){
            count.put(val,count.getOrDefault(val,0)+1);
            //if val not in map assume count=0 and add 1 for every occurrence
        }
        return Collections.max(count.entrySet(), Map.Entry.comparingByValue()).getKey();
        //count.entrySet() → gives all (key, value) pairs.
        //Map.Entry.comparingByValue() → tells max to compare by the frequency, not the key.
        //find the max and return its key
    }

    public int predict(double[] x){
        Node node=root;
        while(!node.isleaf){
            if(x[node.fidx]<node.threshold){
                //value at fidx is less than the threshold
                node=node.left;
            }
            else{
                node=node.right;
            }
        }
        return node.classlabel;
    }

    //predict multiple rows
    public int[] predict(double[][] X){
        int[] predictions=new int[X.length];
        for(int i=0;i<X.length;i++){
            predictions[i]=predict(X[i]);
        }
        return predictions;
    }
        
    //stores the best split
    private class SplitResult{
        int fidx;
        double threshold;
        double[][] Xleft,Xright;
        int[] yleft,yright;

        SplitResult(int fidx,double threshold,double[][] Xleft,int[] yleft,double[][] Xright,int[] yright){
            this.fidx=fidx;
            this.threshold=threshold;
            this.Xleft=Xleft;
            this.yleft=yleft;
            this.Xright=Xright;
            this.yright=yright;
        }
    }

    private SplitResult findbestsplit(double[][] X,int[] y){
        int nSamples=X.length,nFeatures=X[0].length;
        double bestgini=Double.MAX_VALUE;
        SplitResult bestsplit=null;

        //looping through all features n thresholds to get minimized weighted gini impurity
        for(int feature=0;feature<nFeatures;feature++){
            double[] values=new double[nSamples];
            for(int i=0;i<nSamples;i++){
                values[i]=X[i][feature];
            }
            Arrays.sort(values);

            //sorted feature values to try thresholds
            for(double threshold:values){
                List<double[]> leftX=new ArrayList<>();
                List<Integer> leftY=new ArrayList<>();
                List<double[]> rightX=new ArrayList<>();
                List<Integer> rightY=new ArrayList<>();

                for(int i=0;i<nSamples;i++){
                    if(X[i][feature]<threshold){
                        leftX.add(X[i]);
                        leftY.add(y[i]);
                    }
                    else{
                        rightX.add(X[i]);
                        rightY.add(y[i]);
                    }
                }             

                //partition into left n right threshold
                if(leftY.size()==0 || rightY.size()==0){
                    continue;
                }
                //Compute Gini impurity for both sides. Weighted avg impurity = impurity of splits
                double giniLeft = gini(toIntArray(leftY));
                double giniRight = gini(toIntArray(rightY));
                double weighted = (leftY.size() / (double)nSamples) * giniLeft + (rightY.size() / (double)nSamples) * giniRight ;
            
                if(weighted<bestgini){
                    bestgini=weighted;
                    bestsplit=new SplitResult(feature,threshold,leftX.toArray(new double[0][]),toIntArray(leftY),rightX.toArray(new double[0][]),toIntArray(rightY));
                }
            }
        }
        return bestsplit;
    }

    //gini impurity
    private double gini(int[] y){
        Map<Integer,Integer> count=new HashMap<>();
        for(int val:y){
            count.put(val,count.getOrDefault(val,0)+1);
        }
        double impurity=1.0;
        int n=y.length;
        for(int c:count.values()){
            double p=(double)c/n;
            impurity-=p*p;
        }
        return impurity;
    }  

    //helper to convert List<Integer> to int[]
    private int[] toIntArray(List<Integer> list){
        int[] arr=new int[list.size()];
        for(int i=0;i<list.size();i++){
            arr[i]=list.get(i);
        }
        return arr;
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
            return String.format("{\"type\": \"leaf\", \"value\": %d}", node.classlabel);
        } else {
            String leftJson = nodeToJson(node.left);
            String rightJson = nodeToJson(node.right);
            return String.format("{\"type\": \"split\", \"feature_index\": %d, \"threshold\": %.4f, \"left\": %s, \"right\": %s}",
                    node.fidx, node.threshold, leftJson, rightJson);
        }
    }

    public static DecisionTreeClassifier fromJson(Map<String, Object> treeJson) {
        Node root = nodeFromJson(treeJson);
        return new DecisionTreeClassifier(root);
    }

    @SuppressWarnings("unchecked")
    private static Node nodeFromJson(Map<String, Object> nodeJson) {
        String type = (String) nodeJson.get("type");
        if ("leaf".equals(type)) {
            return new Node(((Double) nodeJson.get("value")).intValue());
        } else {
            int featureIndex = ((Double) nodeJson.get("feature_index")).intValue();
            double threshold = (Double) nodeJson.get("threshold");
            Node left = nodeFromJson((Map<String, Object>) nodeJson.get("left"));
            Node right = nodeFromJson((Map<String, Object>) nodeJson.get("right"));
            return new Node(featureIndex, threshold, left, right);
        }
    }
}