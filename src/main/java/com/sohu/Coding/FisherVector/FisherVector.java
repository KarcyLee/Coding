package com.sohu.Coding.FisherVector;

import com.sohu.Coding.MachineLearning.GMM;
import com.sohu.Coding.Tools.Tools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.org.lidalia.sysoutslf4j.context.LogLevel;
import uk.org.lidalia.sysoutslf4j.context.SysOutOverSLF4J;
import weka.core.SerializationHelper;
import java.io.FileNotFoundException;
import java.util.Random;

import static com.sohu.Coding.FisherVector.FisherVectorFlag.FISHER_FLAG_FAST;
import static com.sohu.Coding.FisherVector.FisherVectorFlag.FISHER_FLAG_NORMALIZED;
import static com.sohu.Coding.FisherVector.FisherVectorFlag.FISHER_FLAG_SQUARE_ROOT;
import static java.lang.Math.sqrt;

/**
 * Created by KarlLee on 2016/5/16.
 */
public class FisherVector {
    private static Logger logger = LoggerFactory.getLogger(FisherVector.class);
    private GMM gmm;

    public FisherVector(GMM gmm) {
        this.gmm = gmm;
    }
    public FisherVector(String gmm_path) {
        loadGMM(gmm_path);
    }

    public GMM getGmm() {
        return gmm;
    }
    public void setGmm(GMM gmm) {
        this.gmm = gmm;
    }
    public void loadGMM(String GMM_path){
        try {
            this.gmm = (GMM) SerializationHelper.read(GMM_path);
            logger.info("加载GMM模型成功！");
        }catch (FileNotFoundException fnne){
            logger.error("加载GMM模型失败！"+ GMM_path+ " 不存在！",fnne);
            System.exit(1);
        }catch (Exception e){
            logger.error("加载GMM模型失败！",e);
            System.exit(1);
        }
    }

    public  double[] fisher_encode(double [][]data,int flags){
        if(null == data || data.length == 0 || data[0].length == 0){
            logger.error("pls check the input!");
            return null;
        }
        if(null == gmm){
            logger.error("gmm hadn't be loaded! ");
            return null;
        }
        double [][]means = gmm.getM_Means(); //numClusters * dimension
        double [][] covariances = gmm.getM_Covariances();//numClusters * dimension
        double [] priors = gmm.getM_Priors();//numClusters
        if(null == means || null == covariances || null == priors ){
            logger.error("gmm occurs error! ");
            return null;
        }
        int dimension = means[0].length;
        int numClusters = priors.length;
        int numData = data.length;

        double [] Encoded = new double[ 2 * dimension * numClusters ];//enc

        int dim;
        int i_cl,i_d; //index
        int numTerms = 0;
        double[][] posteriors ;//numClusters * numData
        double[][] sqrtInvSigma;

        assert(numClusters >= 1) ;
        assert(dimension >= 1) ;

        //posteriors = new double[  numClusters * numData ];
        posteriors = new double[numData][numClusters];
        sqrtInvSigma =new double[numClusters][ dimension ];

        for (i_cl = 0 ; i_cl < numClusters ; ++i_cl) {
            for(dim = 0; dim < dimension; dim++) {
                //sqrtInvSigma[i_cl * dimension + dim] = sqrt(1.0 / covariances[i_cl * dimension + dim]);
                sqrtInvSigma[i_cl][dim] = sqrt(1.0 / covariances[i_cl][dim]);
            }
        }


        gmm.get_gmm_data_posteriors(posteriors, numClusters, numData, priors, means, dimension,
                covariances, data) ;
        //Tools.printArray(posteriors);
        /*
        VL_XCAT(vl_get_gmm_data_posteriors_, SFX)(posteriors, numClusters, numData,
                priors,
                means, dimension,
                covariances,
                data) ;
         */

        ///// 稀疏 sparsify posterior assignments with the FAST option

        if ((flags & FISHER_FLAG_FAST) > 0) {
            for(i_d = 0; i_d < numData; i_d ++) {
                //find largest posterior assignment for datum i_d
                int best = 0 ;
                //TYPE bestValue = posteriors[i_d * numClusters] ;
                double bestValue = posteriors[i_d][0] ;
                for (i_cl = 1 ; i_cl < numClusters; ++ i_cl) {
                    //double p = posteriors[i_cl + i_d * numClusters] ;
                    double p = posteriors[i_d][i_cl] ;
                    if (p > bestValue) {
                        bestValue = p ;
                        best = i_cl ;
                    }
                }
                //// make all posterior assignments zero but the best one
                for (i_cl = 0 ; i_cl < numClusters; ++ i_cl) {
                    //posteriors[i_cl + i_d * numClusters] = (double)(i_cl == best) ;
                    if(i_cl != best) {
                        posteriors[i_d][i_cl] = 0;
                    }else{
                        posteriors[i_d ][i_cl] = 1;
                    }
                }
            }
        }
        //Tools.printArray(posteriors);
        for(i_cl = 0; i_cl < numClusters; ++ i_cl) {
            double uprefix;
            double vprefix;

            //double * uk = enc + i_cl*dimension ;
            //double* vk = enc + i_cl*dimension + numClusters * dimension ;
            int idx_uk = i_cl * dimension;
            int idx_vk = i_cl * dimension + numClusters * dimension ;


//        If the GMM component is degenerate and has a null prior, then it
//        must have null posterior as well. Hence it is safe to skip it.  In
//        practice, we skip over it even if the prior is very small; if by
//        any chance a feature is assigned to such a mode, then its weight
//        would be very high due to the division by priors[i_cl] below.

            if (priors[i_cl] < 1e-6) { continue ; }

            for(i_d = 0; i_d < numData; i_d++) {
                //double p = posteriors[i_cl + i_d * numClusters] ;
                double p = posteriors[i_d][i_cl] ;
                System.out.println(p);
                if (p < 1e-6) continue ;
                numTerms += 1;
                for(dim = 0; dim < dimension; ++ dim) {
                    //double diff = data[i_d*dimension + dim] - means[i_cl*dimension + dim] ;
                    double diff = data[i_d][dim] - means[i_cl][dim] ;
                    //diff *= sqrtInvSigma[i_cl*dimension + dim] ;
                    diff *= sqrtInvSigma[i_cl][dim] ;
                    //*(uk + dim) += p * diff ;
                    //*(vk + dim) += p * (diff * diff - 1);
                    Encoded[idx_uk + dim] += p * diff ;
                    Encoded[idx_vk + dim] += p * (diff * diff - 1);
                }
            }

            if (numData > 0) {
                uprefix = 1/(numData * sqrt(priors[i_cl]));
                vprefix = 1/(numData * sqrt(2*priors[i_cl]));
                for(dim = 0; dim < dimension; dim++) {
                    //*(uk + dim) = *(uk + dim) * uprefix;
                    //*(vk + dim) = *(vk + dim) * vprefix;
                    Encoded[idx_uk + dim] *= uprefix;
                    Encoded[idx_vk + dim] *= vprefix;
                }
            }
        }

        //vl_free(posteriors);
        //vl_free(sqrtInvSigma) ;

        // squre_root
        if ((flags & FISHER_FLAG_SQUARE_ROOT) > 0){
            for(dim = 0; dim < 2 * dimension * numClusters ; dim ++) {
                double z = Encoded[ dim ] ;
                if (z >= 0) {
                    Encoded[dim] = sqrt(z) ;
                } else {
                    Encoded[dim] = - sqrt(- z) ;
                }
            }
        }

        //L2归一化
        if ((flags & FISHER_FLAG_NORMALIZED) > 0){
            double n = 0 ;
            for(dim = 0 ; dim < 2 * dimension * numClusters ; dim++) {
                double z = Encoded[dim] ;
                n += z * z ;
            }
            n = sqrt(n) ;
            n = n > 1e-12 ? n : 1e-12;
            for(dim = 0 ; dim < 2 * dimension * numClusters ; dim++) {
                Encoded[dim] /= n ;
            }
        }
        //return numTerms;
        return Encoded;
    }

//    public  int fisher_encode(){
//        int dim;
//        int i_cl,i_d; //index
//        int numTerms = 0;
//        double[][] posteriors ;//numClusters * numData
//        double[][] sqrtInvSigma;
//
//        assert(numClusters >= 1) ;
//        assert(dimension >= 1) ;
//
//        //posteriors = new double[  numClusters * numData ];
//        posteriors = new double[numData][numClusters];
//        sqrtInvSigma =new double[numClusters][ dimension ];
//
//        Encoded = new double[ 2 * dimension * numClusters ];
//
//        for (i_cl = 0 ; i_cl < numClusters ; ++i_cl) {
//            for(dim = 0; dim < dimension; dim++) {
//                //sqrtInvSigma[i_cl * dimension + dim] = sqrt(1.0 / covariances[i_cl * dimension + dim]);
//                sqrtInvSigma[i_cl][dim] = sqrt(1.0 / covariances[i_cl][dim]);
//            }
//        }
//
//
//        /*
//        VL_XCAT(vl_get_gmm_data_posteriors_, SFX)(posteriors, numClusters, numData,
//                priors,
//                means, dimension,
//                covariances,
//                data) ;
//         */
//
//        /* sparsify posterior assignments with the FAST option */
//        //if (flags & VL_FISHER_FLAG_FAST) {
//        if (flags) {
//            for(i_d = 0; i_d < numData; i_d ++) {
//                //find largest posterior assignment for datum i_d
//                int best = 0 ;
//                //TYPE bestValue = posteriors[i_d * numClusters] ;
//                double bestValue = posteriors[i_d][0] ;
//                for (i_cl = 1 ; i_cl < numClusters; ++ i_cl) {
//                    //double p = posteriors[i_cl + i_d * numClusters] ;
//                    double p = posteriors[i_d][i_cl] ;
//                    if (p > bestValue) {
//                        bestValue = p ;
//                        best = i_cl ;
//                    }
//                }
//                //// make all posterior assignments zero but the best one
//                for (i_cl = 0 ; i_cl < numClusters; ++ i_cl) {
//                    //posteriors[i_cl + i_d * numClusters] = (double)(i_cl == best) ;
//                    if(i_cl != best) {
//                        posteriors[i_d][i_cl] = 0;
//                    }else{
//                        posteriors[i_d ][i_cl] = 1;
//                    }
//                }
//            }
//        }
//
//        for(i_cl = 0; i_cl < numClusters; ++ i_cl) {
//            double uprefix;
//            double vprefix;
//
//            //double * uk = enc + i_cl*dimension ;
//            //double* vk = enc + i_cl*dimension + numClusters * dimension ;
//            int idx_uk = i_cl * dimension;
//            int idx_vk = i_cl * dimension + numClusters * dimension ;
//
//
////        If the GMM component is degenerate and has a null prior, then it
////        must have null posterior as well. Hence it is safe to skip it.  In
////        practice, we skip over it even if the prior is very small; if by
////        any chance a feature is assigned to such a mode, then its weight
////        would be very high due to the division by priors[i_cl] below.
//
//            if (priors[i_cl] < 1e-6) { continue ; }
//
//            for(i_d = 0; i_d < numData; i_d++) {
//                //double p = posteriors[i_cl + i_d * numClusters] ;
//                double p = posteriors[i_d][i_cl] ;
//                if (p < 1e-6) continue ;
//                numTerms += 1;
//                for(dim = 0; dim < dimension; dim++) {
//                    //double diff = data[i_d*dimension + dim] - means[i_cl*dimension + dim] ;
//                    double diff = data[i_d][dim] - means[i_cl][dim] ;
//                    //diff *= sqrtInvSigma[i_cl*dimension + dim] ;
//                    diff *= sqrtInvSigma[i_cl][dim] ;
//                    //*(uk + dim) += p * diff ;
//                    //*(vk + dim) += p * (diff * diff - 1);
//                    Encoded[idx_uk + dim] += p * diff ;
//                    Encoded[idx_vk + dim] += p * (diff * diff - 1);
//                }
//            }
//
//            if (numData > 0) {
//                uprefix = 1/(numData*sqrt(priors[i_cl]));
//                vprefix = 1/(numData*sqrt(2*priors[i_cl]));
//                for(dim = 0; dim < dimension; dim++) {
//                    //*(uk + dim) = *(uk + dim) * uprefix;
//                    //*(vk + dim) = *(vk + dim) * vprefix;
//                    Encoded[idx_uk + dim] *= uprefix;
//                    Encoded[idx_vk + dim] *= vprefix;
//                }
//            }
//        }
//
//        //vl_free(posteriors);
//        //vl_free(sqrtInvSigma) ;
//
//        //if (flags & VL_FISHER_FLAG_SQUARE_ROOT) {
//        if (flags){
//        for(dim = 0; dim < 2 * dimension * numClusters ; dim++) {
//                double z = Encoded[dim] ;
//                if (z >= 0) {
//                    Encoded[dim] = sqrt(z) ;
//                } else {
//                    Encoded[dim] = - sqrt(- z) ;
//                }
//            }
//        }
//
//        //if (flags & VL_FISHER_FLAG_NORMALIZED) {
//        if (flags){
//            double n = 0 ;
//            for(dim = 0 ; dim < 2 * dimension * numClusters ; dim++) {
//                double z = Encoded[dim] ;
//                n += z * z ;
//            }
//            n = sqrt(n) ;
//            n = n > 1e-12 ? n : 1e-12;
//            for(dim = 0 ; dim < 2 * dimension * numClusters ; dim++) {
//                Encoded[dim] /= n ;
//            }
//        }
//        return numTerms ;
//    }



    public static void main(String[] args){
        SysOutOverSLF4J.sendSystemOutAndErrToSLF4J(LogLevel.INFO, LogLevel.ERROR);
        Random rd = new Random();
        int nRow = 500;
        int nCol = 100;
        double[][] data= new double[nRow][nCol];
        for(int i = 0; i < nRow;++i ){
            for(int j = 0; j < nCol; ++j ){
                data[i][j] =  rd.nextDouble() * 10;
                //System.out.println(data[i][j]);
            }
        }
        logger.info("数据完毕 ");

        FisherVector fv  = new FisherVector("gmm");
        int flags = FISHER_FLAG_SQUARE_ROOT | FISHER_FLAG_NORMALIZED;
        double [] enc = fv.fisher_encode(data,flags);
        logger.info("打印编码数据 ");
        Tools.printArray(enc);



    }
}
