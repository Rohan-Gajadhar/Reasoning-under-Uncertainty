import java.util.ArrayList;

public class NaiveBayes{
    ArrayList<Instance> trainingData;
    ArrayList<Instance> testData;

    public void print(){
        System.out.println(trainingData);
    }

    public static void main(String[] args){
        String trainingFile = args[0];
        String testFile = args[1];


        NaiveBayes nb = new NaiveBayes();

    }
}