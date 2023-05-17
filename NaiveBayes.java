import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public class NaiveBayes {
    private List<DataRow> trainingData;
    private List<Feature> features;
    private List<String> classLabels;
    private HashMap<String, Integer> classCounts;

    public void loadTrainingData(String path) {
        trainingData = new ArrayList<>();
        classLabels = new ArrayList<String>();
        features = new ArrayList<Feature>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line = br.readLine();
            List<String> featureLabels = Arrays.asList(line.split(","));

            // skip the first two columns, which are instance number and class label
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

                // add all feature values to the set
                for (int i = 0; i < row.getData().size(); i++) {
                    String featureValue = row.getFeatureValue(i);

                    features.get(i).addValue(featureValue);
                }
                // add missing feature values, intialised to 1 as default
                // age feature values
                features.get(0).addValue("10-19");
                features.get(0).addValue("20-29");
                features.get(0).addValue("80-89");
                features.get(0).addValue("90-99");

                // tumor-size feature values
                features.get(2).addValue("55-59");

                // inv-nodes feature values
                features.get(3).addValue("18-20");
                features.get(3).addValue("21-23");
                features.get(3).addValue("27-29");
                features.get(3).addValue("30-32");
                features.get(3).addValue("33-35");
                features.get(3).addValue("36-39");

                trainingData.add(row);
            }
        } catch (Exception e) {
            // handle any I/O problems
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
            // handle any I/O problems
            throw new Error(e);
        }
        return testData;
    }

    String getFeatureValueKey(String featureLabel, String featureValue) {
        return featureLabel + "_" + featureValue;
    }

    String getPerClassFeatureValueKey(String classValue, String featureLabel, String featureValue) {
        return classValue + "_" + featureLabel + "_" + featureValue;
    }

    public HashMap<String, Double> processTrainingData() {
        // total counts for each class
        classCounts = new HashMap<String, Integer>();

        // initial values for increment and any access which doesn't find a value needs to return 1
        HashMap<String, Integer> perClassFeatureCounts = new HashMap<String, Integer>();

        // for each data instance
        for (int i = 0; i < trainingData.size(); i++) {
            DataRow row = trainingData.get(i);

            // increment the class count
            String classLabel = row.getClassLabel();
            Integer newCount = classCounts.getOrDefault(classLabel, 1) + 1;
            classCounts.put(classLabel, newCount);

            for (int f = 0; f < features.size(); f++) {
                Feature feature = features.get(f);
                String featureLabel = feature.getLabel();
                String featureValue = row.getFeatureValue(f);

                String perClassKey = getPerClassFeatureValueKey(classLabel, featureLabel, featureValue);

                // increment the count of occurences of this feature value for this class
                Integer newFeatureCount = perClassFeatureCounts.getOrDefault(perClassKey, 1) + 1;
                perClassFeatureCounts.put(perClassKey, newFeatureCount);
            }
        }

        // calculate the total, or denominators for each feature value
        HashMap<String, Integer> totalCountForFeatureValue = new HashMap<String, Integer>();
        // for each class
        for (int classNum = 0; classNum < classLabels.size(); classNum++) {
            String classLabel = classLabels.get(classNum);

            // for each feature
            for (int f = 0; f < features.size(); f++) {
                Feature feature = features.get(f);
                HashSet<String> featureValues = feature.getValues();

                // for each feature value
                String[] featureValuesArray = featureValues.toArray(new String[featureValues.size()]);
                for (int v = 0; v < featureValuesArray.length; v++) {
                    String featureValue = featureValuesArray[v];

                    String allClassesKey = getFeatureValueKey(feature.getLabel(), featureValue);
                    String perClassKey = getPerClassFeatureValueKey(classLabel, feature.getLabel(), featureValue);

                    // get the current count of each feature value for this class
                    Integer currentValue = totalCountForFeatureValue.getOrDefault(allClassesKey, 0);

                    Integer featureValueCount = perClassFeatureCounts.getOrDefault(perClassKey, 1);
                    currentValue += featureValueCount;
                    totalCountForFeatureValue.put(allClassesKey, currentValue);
                }
            }
        }

        HashMap<String, Double> probabilities = new HashMap<String, Double>();
        // calculate the probabilities
        for (int classNum = 0; classNum < classLabels.size(); classNum++) {
            String classLabel = classLabels.get(classNum);

            // for each feature
            for (int f = 0; f < features.size(); f++) {
                Feature feature = features.get(f);
                HashSet<String> featureValues = feature.getValues();

                // for each feature value
                String[] featureValuesArray = featureValues.toArray(new String[featureValues.size()]);
                for (int v = 0; v < featureValuesArray.length; v++) {
                    String featureValue = featureValuesArray[v];

                    String perClassKey = getPerClassFeatureValueKey(classLabel, feature.getLabel(), featureValue);
                    Double perClassFeatureValueCount = ((double) perClassFeatureCounts.getOrDefault(perClassKey, 1));

                    Double totalClassCount = (double) classCounts.getOrDefault(classLabel, 1);

                    probabilities.put(perClassKey, perClassFeatureValueCount/totalClassCount);
                }
            }
        }
        return probabilities;
    }

    Double predictClassProbability(HashMap<String, Double> probabilities, DataRow row, String classLabel, Double classProbability) {
        Double score = classProbability;
        for (int f = 0; f < features.size(); f++) {
            Feature feature = features.get(f);
            String featureLabel = feature.getLabel();
            String featureValue = row.getFeatureValue(f);

            String perClassKey = getPerClassFeatureValueKey(classLabel, featureLabel, featureValue);
            double perClassProbability = probabilities.getOrDefault(perClassKey, 1.0);
            score *= perClassProbability;
        }
        return score;
    }

    public static void main(String[] args) {
        // command line arguments for training and test files
        /*
        String trainingFile = "";
        String testFile = "";
        if (args.length == 2) {
            trainingFile = args[0];
            testFile = args[1];
        } else {
            System.out.println("Please enter the correct number of arguments");
        }
        */
        String trainingFile = "part2data\\breast-cancer-training.csv";
        String testFile = "part2data\\breast-cancer-test.csv";

        NaiveBayes nb = new NaiveBayes();
        nb.loadTrainingData(trainingFile);
        HashMap<String, Double> probabilities = nb.processTrainingData();

        // print conditional probabilities
        System.out.println("Conditional probabilities:");
        for (Map.Entry<String, Double> entry : probabilities.entrySet()) {
            String[] keyParts = entry.getKey().split("_");
            System.out.println("P(" + keyParts[1] + " = " + keyParts[2] + " | Class = " + keyParts[0] + ") = " + entry.getValue());
        }
        
        // print class probabilities
        System.out.println("\nClass probabilities:");
        for (Map.Entry<String, Integer> entry : nb.classCounts.entrySet()) {
            System.out.println("P(Y = " + entry.getKey() + ") = " + ((double) entry.getValue()) / nb.trainingData.size());
        }
        System.out.println("\n");

        var testData = nb.loadTestData(testFile);

        int correct = 0;
        int testInstanceNumber = 0;
        for (DataRow row : testData) {
            Double highestProbability = 0.0;
            String predictedClass = "";
            for (String label : nb.classLabels) {
                Double classProbability = ((double) nb.classCounts.getOrDefault(label, 0)) / nb.trainingData.size();
                Double probability = nb.predictClassProbability(probabilities, row, label, classProbability);
                System.out.println("Instance: " + testInstanceNumber + "   Score(Class = " + label + ") = " + probability);
                if (probability > highestProbability) {
                    predictedClass = label;
                    highestProbability = probability;
                }
                
            }
            if (predictedClass.equals(row.getClassLabel())) {
                correct++;
            }
            System.out.println("Predicted class: " + predictedClass + "\n");
            testInstanceNumber++;
        }
        System.out.println("Testing accuracy: " + ((double) correct / testData.size()) * 100 + "%");
    }
}

class DataRow {
    private List<String> data;
    private String instanceNumber;
    private String classLabel;

    public DataRow(String instanceNumber, String classLabel, List<String> data) {
        this.instanceNumber = instanceNumber;
        this.data = data;
        this.classLabel = classLabel;
    }

    public String getInstanceNumber() {
        return instanceNumber;
    }

    public List<String> getData() {
        return data;
    }

    public String getFeatureValue(int index) {
        return data.get(index);
    }

    public String getClassLabel() {
        return classLabel;
    }

    public String toString() {
        return instanceNumber + " " + classLabel + " " + data.toString();
    }
}

class Feature {
    private String label;
    private HashSet<String> values;

    public Feature(String label) {
        this.label = label;
        values = new HashSet<String>();
    }

    public HashSet<String> getValues() {
        return values;
    }

    public String getLabel() {
        return label;
    }

    public void addValue(String value) {
        values.add(value);
    }
}