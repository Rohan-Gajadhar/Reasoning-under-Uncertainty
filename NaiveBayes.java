import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class NaiveBayes{
    private List<String[]> trainingData;
    
    public List<String[]> getData(String path) {
        trainingData = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] lineItems = line.split(",");
                trainingData.add(lineItems);
            }
        } catch (Exception e) {
            // Handle any I/O problems
            throw new Error(e);
        }

        return trainingData;
    }

    public List<String[]> getClassLabels(){
        List<String[]> classLabels = new ArrayList<>();
        //for the first row in trainingdata
        //add to classLabels
        String[] line = trainingData.get(0);
        classLabels.add(line);

        /*
        Print out the class labels
        for(int i = 0; i < classLabels.size(); i++){
            String[] lineItems = classLabels.get(i);
            for(int j = 0; j < lineItems.length; j++){
                System.out.print(lineItems[j] + " ");
            }
            System.out.println();
        }
        */

        return classLabels;
    }
    
    public void processData(){
        /* 
        Print out the data
        for(int i = 0; i < trainingData.size(); i++){
            String[] line = trainingData.get(i);
            for(int j = 0; j < line.length; j++){
                System.out.print(line[j] + " ");
            }
            System.out.println();
        }
        */
        int count = 0;
        //for each class label
        
    }

    public static void main(String[] args){
        /* Cmd line file path args:
        String trainingFile = args[0];
        String testFile = args[1];
        */

        String trainingFile = "Assignment3\\part2data\\breast-cancer-training.csv";
        //String testFile = "breast-cancer-test.csv";

        NaiveBayes nb = new NaiveBayes();
        nb.getData(trainingFile);
        nb.processData();
        nb.getClassLabels();
    }
}