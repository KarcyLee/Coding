package com.sohu.Coding.FisherVector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

import static java.lang.Math.sqrt;

/**
 * Created by pengli211286 on 2016/5/16.
 */
public class FisherVector {
    private static Logger logger = LoggerFactory.getLogger(FisherVector.class);

    int dimension;
    int numClusters;
    double [][]means; //numClusters * dimension
    double [][] covariances;//numClusters * dimension
    double [] priors;//numClusters
    int numData;
    double [][]data ;//toBeEncoded;
    boolean flags;
    double [] Encoded;//enc

    public  int fisher_encode(){
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

        Encoded = new double[ 2 * dimension * numClusters ];

        for (i_cl = 0 ; i_cl < numClusters ; ++i_cl) {
            for(dim = 0; dim < dimension; dim++) {
                //sqrtInvSigma[i_cl * dimension + dim] = sqrt(1.0 / covariances[i_cl * dimension + dim]);
                sqrtInvSigma[i_cl][dim] = sqrt(1.0 / covariances[i_cl][dim]);
            }
        }
        /*
        VL_XCAT(vl_get_gmm_data_posteriors_, SFX)(posteriors, numClusters, numData,
                priors,
                means, dimension,
                covariances,
                data) ;
         */

        /* sparsify posterior assignments with the FAST option */
        //if (flags & VL_FISHER_FLAG_FAST) {
        if (flags) {
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
                if (p < 1e-6) continue ;
                numTerms += 1;
                for(dim = 0; dim < dimension; dim++) {
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
                uprefix = 1/(numData*sqrt(priors[i_cl]));
                vprefix = 1/(numData*sqrt(2*priors[i_cl]));
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

        //if (flags & VL_FISHER_FLAG_SQUARE_ROOT) {
        if (flags){
        for(dim = 0; dim < 2 * dimension * numClusters ; dim++) {
                double z = Encoded[dim] ;
                if (z >= 0) {
                    Encoded[dim] = sqrt(z) ;
                } else {
                    Encoded[dim] = - sqrt(- z) ;
                }
            }
        }

        //if (flags & VL_FISHER_FLAG_NORMALIZED) {
        if (flags){
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
        return numTerms ;
    }

}
