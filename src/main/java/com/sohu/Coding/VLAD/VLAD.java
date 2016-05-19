package com.sohu.Coding.VLAD;

import com.sohu.MachineLearning.KMEANS;

import com.sohu.Tools.Tools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.org.lidalia.sysoutslf4j.context.LogLevel;
import uk.org.lidalia.sysoutslf4j.context.SysOutOverSLF4J;
import weka.core.SerializationHelper;

import java.io.Serializable;
import java.util.Random;

import static com.sohu.Coding.VLAD.VLADFlag.*;

/**
 * Created by KarlLee on 2016/5/18.
 */
public class VLAD implements Serializable, Cloneable{
    private static final long serialVersionUID = 1969698442195388519L;
    private static Logger logger = LoggerFactory.getLogger(VLAD.class);
    private static String ClassInfo = "Vector of Locally Aggregated Descriptors (VLAD) encoding ";

    private KMEANS kmeans;//必须为已经训练好的KMeans，包含中心向量
    private int flags = VLAD_FLAG_SQUARE_ROOT |VLAD_FLAG_NORMALIZE_COMPONENTS ;

    public VLAD(KMEANS kmeans) {
        this.kmeans = kmeans;
    }
    public VLAD(String strKmeans){
        loadKMEANS(strKmeans);
    }

    public void setFlags(int flags) {
        this.flags = flags;
    }
    public int getFlags() {

        return flags;
    }
    public KMEANS getKmeans() {
        return kmeans;
    }
    public void setKmeans(KMEANS kmeans) {

        this.kmeans = kmeans;
    }

    public void loadKMEANS(String strKmeans){
        try {
            //加载
            KMEANS km = (KMEANS) SerializationHelper.read("Kmeans");
            this.kmeans = km;
        }catch (Exception e){
            logger.error("加载KMeans失败！",e);
            System.exit(1);
        }
    }



    public double[] vlad_encode(double [][]data){
        if(null == data || data.length == 0 || data[0].length == 0){
            logger.error("pls check the input!");
            return null;
        }
        if(null == kmeans){
            logger.error("KMEANS hadn't be loaded! ");
            return null;
        }

        double [][] means = kmeans.getCenter();
        if(null == means || means.length == 0 || means[0].length == 0){
            logger.error("the center of kmeans is WRONG！ ");
            return null;
        }
        int dimension = means[0].length;
        int numCenters = means.length;
        int numData = data.length;
        if(dimension != data[0].length){
            logger.error("the dimension doesn't match！");
            return null;
        }
        //assert(dimension == data[0].length);


        // find nearest cliuster centers for the data that should be encoded
        int [] indexes =  this.kmeans.Quantize(data);
        // convert indexes array to assignments array,which can be processed by vlad_encode
        double []assignments = new double[numData * numCenters];// numDataToEncode * numCenters
        for(int i = 0; i < numData; ++ i ) {
            assignments[i * numCenters + indexes[i]] = 1.;
        }
        //////////////////////////////////////////////////////////
        double []enc = new double[dimension * numCenters];
        int dim ;
        int i_cl, i_d ;

        for (i_cl = 0; i_cl < numCenters; ++i_cl) {
            double clusterMass = 0 ;
            for (i_d = 0; i_d < numData; ++i_d) {
                if (assignments[ i_d * numCenters + i_cl] > 0) {
                    double q = assignments[ i_d * numCenters + i_cl] ;
                    clusterMass +=  q ;
                    for(dim = 0; dim < dimension; dim++) {
                       // enc [i_cl * dimension + dim] += q * data [i_d  * dimension + dim] ;
                        enc [i_cl * dimension + dim] += q * data [i_d ][dim] ;
                    }
                }
            }

            if (clusterMass > 0) {
                if ((flags & VLAD_FLAG_NORMALIZE_MASS) > 0) {
                    //Each vector Vk is divided by the total mass of feature_vectors associated to it
                    for(dim = 0; dim < dimension; ++dim) {
                        enc[i_cl * dimension + dim] /= clusterMass ;
                        //enc[i_cl*dimension + dim] -= means[i_cl*dimension+dim];
                        enc[i_cl * dimension + dim] -= means[i_cl][dim];
                    }
                } else {
                    for(dim = 0; dim < dimension; ++dim) {
                        //enc[i_cl*dimension + dim] -= clusterMass * means[i_cl*dimension+dim];
                        enc[i_cl*dimension + dim] -= clusterMass * means[i_cl][dim];
                    }
                }
            }

            //squre root : sign(z)*sqrt(z)
            if ((flags & VLAD_FLAG_SQUARE_ROOT) > 0) {
                for(dim = 0; dim < dimension; ++ dim) {
                    double z = enc[i_cl*dimension + dim] ;
                    if (z >= 0) {
                        enc[i_cl*dimension + dim] = Math.sqrt(z) ;
                    } else {
                        enc[i_cl*dimension + dim] = - Math.sqrt(- z) ;
                    }
                }
            }

            // 对每个子向量进行L2归一化
            if ((flags & VLAD_FLAG_NORMALIZE_COMPONENTS) > 0) {
                double n = 0 ;
                for(dim = 0; dim < dimension; ++dim) {
                    double z = enc[ i_cl * dimension + dim] ;
                    n += z * z ;
                }
                n = Math.sqrt(n) ;
                n = Math.max(n, 1e-12) ;
                for(dim = 0; dim < dimension; ++dim) {
                    enc[i_cl*dimension + dim] /= n ;
                }
            }
        }

        //对整个进行L2归一化
        if ((flags & VLAD_FLAG_NORMALIZE_GLOBAL) > 0) {
            double n = 0 ;
            for( dim = 0; dim < enc.length; ++dim) {
                double z = enc[  dim] ;
                n += z * z ;
            }
            n = Math.sqrt(n) ;
            n = Math.max(n, 1e-12) ;
            for( dim = 0; dim < enc.length; ++dim) {
                enc[dim] /= n ;
            }
        }

        return enc;
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
            }
        }
        logger.info("数据完毕 ");


        try {
            //加载
            KMEANS km = (KMEANS) SerializationHelper.read("Kmeans");
            VLAD vlad = new VLAD(km);
            vlad.setFlags( VLAD_FLAG_NORMALIZE_MASS);
            double[] enc = vlad.vlad_encode(data);
            Tools.printArray(enc);
        }catch (Exception e){
            logger.error("",e);
        }



    }

}
