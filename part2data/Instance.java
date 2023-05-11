import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class Instance {
    public static List<String[]> getLines(String path) {
        List<String[]> rowList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] lineItems = line.split(",");
                rowList.add(lineItems);
            }
        } catch (Exception e) {
            // Handle any I/O problems
            throw new Error(e);
        }

        return rowList;
    }

    public static String[][] getData(List<String[]> rows) {
        String[][] data = new String[rows.size()][];
        for (int i = 0; i < rows.size(); i++) {
            String[] row = rows.get(i);
            //Exclude label row
            for (int j = 0; j < row.length - 1; j++){
                data[i][j] = row[j];
            }
        }
        return data;
    }

    public static String[] getLabels(List<String[]> rows) {
        String[] labels = new String[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            String[] row = rows.get(i);
            labels[i] = row[row.length - 1];
        }
        return labels;
    }
}
