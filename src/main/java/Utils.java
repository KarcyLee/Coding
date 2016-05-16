/**
 * Created by pengli211286 on 2016/5/16.
 */
public class Utils {
    double [] mean(double [][]data){
        if(null == data)
            return null;
        int nRow = data.length;
        if(nRow == 0)
            return null;
        int nCol = data[0].length;
        double[] mean = new double[nCol];
        for(int j = 0; j < nCol;++j){
            for(int i = 0; i < nRow;++ i){
                mean[j] += data[i][j];
            }
            mean[j] /= nRow;
        }
        return mean;
    }
}
