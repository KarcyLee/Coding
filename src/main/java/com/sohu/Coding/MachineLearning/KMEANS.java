package com.sohu.Coding.MachineLearning;


import com.sohu.Coding.Tools.Distance;
import com.sohu.Coding.Tools.Tools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.org.lidalia.sysoutslf4j.context.LogLevel;
import uk.org.lidalia.sysoutslf4j.context.SysOutOverSLF4J;
import weka.core.SerializationHelper;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.*;

/**
 * Created by KarlLee on 2016/5/18.
 */
public class KMEANS implements Serializable, Cloneable {
    private static final long serialVersionUID = 5387925096350970389L;
    private static Logger logger = LoggerFactory.getLogger(KMEANS.class);

    int numCenters = 100; // 指定划分的簇数
    double mu = 1e-5; // 迭代终止条件，当各个新质心相对于老质心偏移量小于mu时终止迭代
    int maxIter = 1000;
    int curIter = 0;
    double[][] center; // 上一次各簇质心的位置
    double [][] data;//存放待聚类数据
    int [] assign; //存放每个数据所属的类别。numData

    public KMEANS() {

    }
    public KMEANS(int k, double mu, int maxIter) {
        this.numCenters = k;
        this.mu = mu;
        this.maxIter = maxIter;
    }
    public KMEANS(double[][] data,int k,double mu,int maxIter) {
        this.numCenters = k;
        this.mu = mu;
        this.data = data;
        this.maxIter = maxIter;
    }

    public void setData(double[][] data) {
        this.data = data;
    }
    public void setNumCenters(int k) {
        this.numCenters = k;
    }
    public int getNumCenters() {
        return numCenters;
    }
    public void setMu(double mu) {
        this.mu = mu;
    }
    public void setMaxIter(int maxIter) {
        this.maxIter = maxIter;
    }
    public double[][] getCenter() {
        return center;
    }
    public int getMaxIter() {
        return maxIter;
    }
    public double getMu() {
        return mu;
    }
    public int[] getAssign() {
        return assign;
    }
    public double[][] getData() {
        return data;
    }

    // 初始化k个质心，每个质心是len维的向量，每维均在left--right之间
    public void initCenter( ) {
        if(null ==  data || data.length == 0 || data[0].length == 0) {
            logger.error("空数据！");
            System.exit(1);
        }
        if( this.numCenters < 0 ){
            logger.error("聚类中心数目错误！");
            System.exit(1);
        }

        Random random = new Random(System.currentTimeMillis());
        int[] count = new int[numCenters]; // 记录每个簇有多少个元素

        int numData = data.length;
        int dimension = data[0].length;
        this.center = new double[numCenters][dimension];
        this.assign = new int[numData];


        for(int i_d = 0; i_d < numData; ++i_d){
            int i_c = random.nextInt(10000) % numCenters;
            count[i_c]++;
            Tools.arrayAddEqual(center[i_c],data[i_d]);
            assign[i_d] = i_c;
        }
        for (int i = 0; i < numCenters; ++i) {
            if(count[i] != 0){
                for (int j = 0; j < dimension; ++j) {
                    center[i][j] /= count[i];
                }
            }else{
                // 簇中不包含任何点,及时调整质心
                for (int j = 0; j < dimension; ++j) {
                    center[i][j] = random.nextDouble();
                }
            }
        }
    }

    // 把数据集中的每个点归到离它最近的那个质心
    public void classify( ) {

        int numData = data.length;
        int dimension = data[0].length;

        //统计新的分布
        for(int i_d = 0; i_d < numData; ++ i_d){
            int index = 0; // 最近中心的类别
            double nearDist = Double.MAX_VALUE;
            for(int i_c = 0; i_c < numCenters; ++ i_c){
                double dist = Distance.CosineDistance(data[i_d],center[i_c]);
                if( dist < nearDist){
                    index = i_c;
                    nearDist = dist;
                }
            }
            this.assign[i_d] = index;
        }

    }

    // 重新计算每个簇的质心，并判断终止条件是否满足，如果不满足更新各簇的质心,如果满足就返回true.len是数据的维数
    public boolean calNewCenter() {
        boolean end = true;

        int k = this.numCenters;
        int[] count = new int[k]; // 记录每个簇有多少个元素
        double satisfy = 0.0; // 满意度
        double[] ss = new double[k]; //满意度数组

        int numData = data.length;
        int dimension = data[0].length;
        double[][] sum = new double[k][dimension];

        for(int i_d = 0; i_d < numData; ++ i_d){
            int i_c = assign[i_d]; //聚类中心的类别
            Tools.arrayAddEqual(sum[i_c],data[i_d]);
            ++ count[i_c];
        }
        //均值求质心
        for (int i = 0; i < k; i++) {
            if (count[i] != 0) {
                for (int j = 0; j < dimension; j++) {
                    sum[i][j] /= count[i];
                }
            }
            // 簇中不包含任何点,及时调整质心
            else {
                int a= (i + 1) % k;
                int b=(i + 3) % k;
                int c=(i + 5) % k;
                for (int j = 0; j < dimension; j++) {
                    center[i][j] = (center[a][j] + center[b][j] + center[c][j])/3;
                }
            }
        }
        for (int i = 0; i < k; i++) {
            // 只要有一个质心需要移动的距离超过了mu，就返回false
            if (Distance.CosineDistance(sum[i], center[i]) >= mu){
                end = false;
                break;
            }
        }
        if (! end) {
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < dimension; j++)
                    center[i][j] = sum[i][j];
            }
        }
        return end;
    }

    // 计算各簇内数据和方差的加权平均，得出本次聚类的满意度.len是数据的维数
    public double getSati( ) {
        int k = this.numCenters;
        int numData = data.length;
        int dimension = data[0].length;
        double satisfy = 0.0;
        int[] count = new int[k];
        double[] ss = new double[k];

        for(int i_d = 0; i_d < numData; ++ i_d){
            int i_c = assign[i_d]; //聚类中心的类别
            for(int i = 0; i < dimension; ++ i){
                double a = this.data[i_d][i] - center[i_c][i];
                ss[i_c] += (a * a);
            }
            ++ count[i_c];
        }
        for (int i = 0; i < k; i++) {
            satisfy += count[i] * ss[i];
        }
        return satisfy;
    }

    public void cleanUp(){
        this.data = null;
        this.assign = null;
    }


    public double runCluster( ) {
        if(null ==  data || data.length == 0 || data[0].length == 0) {
            logger.error("空数据！");
            System.exit(1);
        }
        if( this.numCenters <= 0 ){
            logger.error("聚类中心数目错误！");
            System.exit(1);
        }

        initCenter( );
        classify( );
        ++ curIter;
        while (! calNewCenter( )) {
             classify( );
             ++curIter;
            logger.info("第"+ Integer.toString(curIter)+ "次迭代");
            if(curIter >= this.maxIter) {
                break;
            }
        }

        double ss = getSati( );
        System.out.println("加权方差：" + ss);
        return ss;
    }


//    vl_uint32 * assignments = vl_malloc(sizeof(vl_uint32) * numData) ;
//    float * distances = vl_malloc(sizeof(float) * numData) ;
//    vl_kmeans_quantize(kmeans, assignments, distances, data, numData) ;
/* ---------------------------------------------------------------- */
/*                                                     Quantization */
/* ---------------------------------------------------------------- */

   public  int[] Quantize( double[][] data){
       //均采用余弦距离
       if(null == data || data.length == 0 || data[0].length == 0){
           logger.error("pls check the input!");
           return null;
       }
       int numData = data.length;
       int dimension = data[0].length;
       if(null == this.center  || this.center.length == 0
               || this.center[0].length == 0 || this.center[0].length != dimension ){
           logger.error("聚类中心有误！请重新聚类！");
           return null;
       }
       //确保k == numCenters
       this.numCenters = this.center.length;
       int []assignments = new int[numData];

       double []distanceToCenters = new double[numCenters];

       for (int i_d = 0 ; i_d < numData ; ++ i_d) {
           double bestDistance = Double.MAX_VALUE ;
           for(int i_c = 0; i_c < this.numCenters; ++ i_c){
               distanceToCenters[i_c] = Distance.CosineDistance(data[i_d],this.center[i_c]);
           }
           for (int k = 0 ; k < this.numCenters ; ++k) {
               if (distanceToCenters[k] < bestDistance) {
                   bestDistance = distanceToCenters[k] ;
                   assignments[i_d] = k ;
               }
           }
//           if (distances)
//               distances[i] = bestDistance ;
       }
       return assignments;
   }
    public int Quantize( double[] data){
        //均采用余弦距离
        if(null == data || data.length == 0 ){
            logger.error("pls check the input!");
            return -1;
        }
        int dimension = data.length;
        if(null == this.center  || this.center.length == 0
                || this.center[0].length == 0 || this.center[0].length != dimension ){
            logger.error("聚类中心有误！请重新聚类！");
            return -1;
        }
        //确保k == numCenters
        this.numCenters = this.center.length;
        int assignments = 0;
        double []distanceToCenters = new double[numCenters];
        double bestDistance = Double.MAX_VALUE ;
        for(int i_c = 0; i_c < this.numCenters; ++ i_c){
            distanceToCenters[i_c] = Distance.CosineDistance(data,this.center[i_c]);
        }
        for (int k = 0 ; k < this.numCenters ; ++k) {
            if (distanceToCenters[k] < bestDistance) {
                bestDistance = distanceToCenters[k] ;
                assignments = k ;
            }
        }
        return assignments;
    }





    private void writeObject(ObjectOutputStream out) throws IOException {
        cleanUp();
        out.defaultWriteObject();
    }
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException{
        //完成默认的反序列化读取
        in.defaultReadObject();
    }

    public static void main(String[] args) {
        SysOutOverSLF4J.sendSystemOutAndErrToSLF4J(LogLevel.INFO, LogLevel.ERROR);

        Random rd = new Random();
        int nRow = 1000;
        int nCol = 100;
        double[][] data= new double[nRow][nCol];
        for(int i = 0; i < nRow;++i ){
            for(int j = 0; j < nCol; ++j ){
                data[i][j] =  rd.nextDouble() * 10;
                //System.out.println(data[i][j]);
            }
        }
        logger.info("数据完毕 ");


        KMEANS km = new KMEANS(data,5,1e-6,2000);
        km.runCluster();
        try {
            //保存
            SerializationHelper.write("Kmeans", km);
            //加载
            km = (KMEANS) SerializationHelper.read("Kmeans");
            double[][] center = km.getCenter();
            Tools.printArray(center);
        }catch (Exception e){
            logger.error("",e);
        }
    }


}
