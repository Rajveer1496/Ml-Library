package core.utils;

public class Mathutils {
    //overloading use
    public double mean_cal(double[] nums){
        int n=nums.length;
        double sum=0;
        for(int i=0;i<nums.length;i++){
            sum+=nums[i];
        }
        return (sum/(double)n);
    }

    public int mean_cal(int[] nums){
        int sum=0;
        int n=nums.length;
        for(int i=0;i<n;i++){
            sum+=nums[i];
        }
        return sum/n;
    }
    //  POWER 
    public double power(double base,int power){
        double ans=1.0;
        for(int i=0;i<power;i++){
            ans*=base;
        }
        return ans;
    }
    //SQUARE ROOT
    public double sqrt(double num){
        if(num<0){
            return -1;
        }
        double guess=num/2.0;
        for(int i=0;i<20;i++){
            guess=(guess+num/guess)/2.0;
        }
        return guess;
    }
    //VARIANCE
    public double variance_cal(double[] nums){
        double mean=mean_cal(nums);
        int n=nums.length;
        double sum=0;
        for(int i=0;i<n;i++){
            double diff=nums[i]-mean;
            sum=sum+power(diff,2);
        }
        return (sum/(double)n);
    }
    //STANDARD DEVIATION
    public double std_dev(double[] nums){
        return sqrt(variance_cal(nums));
    }
    //DOT PRODUCT
    public double dot_product(double[] num1,double[] num2){
        double sum=0;
        int l1=num1.length;
        int l2=num2.length;
        int min=l1;
        if( l1 >= l2){
            min=l2;                  //we won't consider the other elements left since it will turn 0
        }
         for(int i=0;i<min;i++){
            sum=sum+(num1[i]*num2[i]);
        }
        
        return sum;
    }
    //MATRIX TRANSPOSE
    public double[][] transpose(double[][] mat){
        int r=mat[0].length;
        int c=mat[0].length;
        double[][] ans=new double[c][r];
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                ans[j][i]=mat[i][j];
            }
        }
        return ans;
    }

    //CUMULATIVE ARRAY
    public int[] cumulative(int[] nums){
        int arr[]=new int[nums.length];
        arr[0]=nums[0];
        for(int i=1;i<nums.length;i++){
            arr[i]=nums[i]+nums[i-1];
        }
        return arr;
    }
}
