import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

class DataRow{
    private List<String> data;
    private String instanceNumber;
    private String classLabel;

    public DataRow(String instanceNumber, String classLabel, List<String> data){
        this.instanceNumber = instanceNumber;
        this.data = data;
        this.classLabel = classLabel;
    }

    public String getInstanceNumber(){
        return instanceNumber;
    }

    public List<String> getData(){
        return data;
    }

    public String getFeatureValue(int index){
        return data.get(index);
    }

    public String getClassLabel(){
        return classLabel;
    }

    public String toString(){
        return instanceNumber + " " + classLabel + " " + data.toString();
    }
}

class Feature{
    private String label;
    private HashSet<String> values;

    public Feature(String label){
        this.label = label;
        values = new HashSet<String>();
    }

    public HashSet<String> getValues(){
        return values;
    }

    public String getLabel(){
        return label;
    }

    public void addValue(String value){
        values.add(value);
    }
}

public class NaiveBayes{
    private List<DataRow> trainingData;
    private List<Feature> features;
    private List<String> classLabels;
    
    public void loadTrainingData(String path) {
        trainingData = new ArrayList<>();
        classLabels = new ArrayList<String>();
        features = new ArrayList<Feature>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line = br.readLine();
            List<String> featureLabels =  Arrays.asList(line.split(","));

            //skip the first two columns, which are instance number and class label
            for (String label : featureLabels.subList(2, featureLabels.size())) {
                features.add(new Feature(label));
            }

            while ((line = br.readLine()) != null) {
                List<String> lineItems = Arrays.asList(line.split(","));

                String classLabel = lineItems.get(1);
                if (!classLabels.contains(classLabel)) {
                    classLabels.add(classLabel);
                }
                DataRow row = new DataRow(lineItems.get(0), classLabel, lineItems.subList(2, lineItems.size()));

                //add all feature values to the set
                for(int i = 0; i < row.getData().size(); i++){
                    String featureValue = row.getFeatureValue(i);

                    features.get(i).addValue(featureValue);
                }
                trainingData.add(row);
            }
        } catch (Exception e) {
            // Handle any I/O problems
            throw new Error(e);
        }
    }

    public List<DataRow> loadTestData(String path) {
        var testData = new ArrayList<DataRow>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line = br.readLine();

            while ((line = br.readLine()) != null) {
                List<String> lineItems = Arrays.asList(line.split(","));

                String classLabel = lineItems.get(1);
                DataRow row = new DataRow(lineItems.get(0), classLabel, lineItems.subList(2, lineItems.size()));

                testData.add(row);
            }
        } catch (Exception e) {
            // Handle any I/O problems
            throw new Error(e);
        }
        return testData;
    }

    String getFeatureValueKey(String featureLabel, String featureValue){
        return featureLabel + "_" + featureValue;
    }

    String getPerClassFeatureValueKey(String classValue, String featureLabel, String featureValue){
        return classValue + "_" + featureLabel + "_" + featureValue;
    }

    public HashMap<String, Double> processTrainingData(){
        //total counts for each class
        HashMap<String, Integer> classCounts = new HashMap<String, Integer>();

        //initial values for increment and any access which doesn't find a value needs to return 1
        HashMap<String, Integer> perClassFeatureCounts = new HashMap<String, Integer>();

        //for each data instance
        for(int i = 0; i < trainingData.size(); i++){
            DataRow row = trainingData.get(i);

            //increment the class count
            String classLabel = row.getClassLabel();
            Integer newCount = classCounts.getOrDefault(classLabel, 0) + 1;
            classCounts.put(classLabel, newCount);

            for(int f = 0; f < features.size(); f++){
                Feature feature = features.get(f);
                String featureLabel = feature.getLabel();
                String featureValue = row.getFeatureValue(f);

                String perClassKey = getPerClassFeatureValueKey(classLabel, featureLabel, featureValue);

                //increment the count of occurences of this feature value for this class
                Integer newFeatureCount = perClassFeatureCounts.getOrDefault(perClassKey, 1) + 1;
                perClassFeatureCounts.put(perClassKey, newFeatureCount);
            }
        }

        //calculate the total, or denominators for each feature value
        HashMap<String, Integer>  totalCountForFeatureValue = new HashMap<String, Integer>();
        //for each class
        for(int classNum = 0; classNum < classLabels.size(); classNum++){
            String classLabel = classLabels.get(classNum);

            //for each feature
            for(int f = 0; f < features.size(); f++){
                Feature feature = features.get(f);
                HashSet<String> featureValues = feature.getValues();

                //for each feature value
                String[] featureValuesArray = featureValues.toArray(new String[featureValues.size()]);
                for(int v = 0; v < featureValuesArray.length; v++){
                    String featureValue = featureValuesArray[v];

                    String allClassesKey = getFeatureValueKey(feature.getLabel(), featureValue);
                    String perClassKey = getPerClassFeatureValueKey(classLabel, feature.getLabel(), featureValue);

                    //get the current count of each feature value for this class
                    Integer currentValue = totalCountForFeatureValue.getOrDefault(allClassesKey, 0);

                    Integer featureValueCount = perClassFeatureCounts.getOrDefault(perClassKey, 1);
                    currentValue += featureValueCount;
                    totalCountForFeatureValue.put(allClassesKey, currentValue);
                }
            }
        }

        HashMap<String, Double> probabilities = new HashMap<String, Double>();
        //calculate the probabilities
        for(int classNum = 0; classNum < classLabels.size(); classNum++){
            String classLabel = classLabels.get(classNum);

            //for each feature
            for(int f = 0; f < features.size(); f++){
                Feature feature = features.get(f);
                HashSet<String> featureValues = feature.getValues();

                //for each feature value
                String[] featureValuesArray = featureValues.toArray(new String[featureValues.size()]);
                for(int v = 0; v < featureValuesArray.length; v++){
                    String featureValue = featureValuesArray[v];

                    String allClassesKey = getFeatureValueKey(feature.getLabel(), featureValue);
                    String perClassKey = getPerClassFeatureValueKey(classLabel, feature.getLabel(), featureValue);
                    Double perClassFeatureCount = ((double)perClassFeatureCounts.getOrDefault(perClassKey, 1));

                    Double totalFeatureCount = ((double)totalCountForFeatureValue.getOrDefault(allClassesKey, 1));

                    probabilities.put(perClassKey, 
                        perClassFeatureCount / 
                        totalFeatureCount);
                }
            }
        }
        return probabilities;
    }

    Double predictClassProbability(HashMap<String, Double> probabilities, DataRow row, String classLabel){
        Double probability = 1.0;
        for(int f = 0; f < features.size(); f++){
            Feature feature = features.get(f);
            String featureLabel = feature.getLabel();
            String featureValue = row.getFeatureValue(f);

            String perClassKey = getPerClassFeatureValueKey(classLabel, featureLabel, featureValue);
            probability *= probabilities.getOrDefault(perClassKey, 1.0);
        }
        return probability;
    }

    public static void main(String[] args){
        /* Cmd line file path args:
        String trainingFile = args[0];
        String testFile = args[1];
        */

        String trainingFile = "part2data/breast-cancer-training.csv";
        String testFile = "part2data/breast-cancer-test.csv";

        NaiveBayes nb = new NaiveBayes();
        nb.loadTrainingData(trainingFile);
        var probabilities = nb.processTrainingData();

        //print probabilities
        /*
        System.out.println("Probabilities:");
        for (Map.Entry<String, Double> entry : probabilities.entrySet()) {
            System.out.println(entry.getKey() + " = " + entry.getValue());
        }
        */

        var testData = nb.loadTestData(testFile);

        for(DataRow row : testData){
            Double highestProbability = 0.0;
            String predictedClass = "";
            for(String label : nb.classLabels){
                Double probability = nb.predictClassProbability(probabilities, row, label);
                System.out.println("Instance: " + row.getInstanceNumber() + " probability of " + label + " = " + probability);
                if (probability > highestProbability){
                    predictedClass = label;
                    highestProbability = probability;
                }
            }
            System.out.println("Predicted label of: " + predictedClass);
        } 
    }
}