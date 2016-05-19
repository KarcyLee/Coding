package com.sohu.MachineLearning;


import com.sohu.Tools.Tools;
import org.apache.commons.math3.distribution.MixtureMultivariateNormalDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.fitting.MultivariateNormalMixtureExpectationMaximization;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.org.lidalia.sysoutslf4j.context.LogLevel;
import uk.org.lidalia.sysoutslf4j.context.SysOutOverSLF4J;
import weka.core.SerializationHelper;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * Created by KarlLee on 2016/5/16.
 */
////GaussianMixtureModel
public class GMM implements Serializable, Cloneable{
    private static final long serialVersionUID = 2361103170130142208L;
    private static Logger logger = LoggerFactory.getLogger(GMM.class);

    double [][]m_Means; //numClusters * dimension
    double [][] m_Covariances;//numClusters * dimension
    double [] m_Priors;//numClusters, weights


    public double[][] getM_Means() {
        return m_Means;
    }

    public double[] getM_Priors() {
        return m_Priors;
    }

    public double[][] getM_Covariances() {
        return m_Covariances;
    }

    public void runEM(double[][] data, int numCluster, int numMaxIter){
        try{
            int K = numCluster;
            //数据归一化
            //data= Tools.normalizeFeatures(data);
            logger.info("打印训练数据");
            Tools.printArray(data);
            int V= data[0].length; //dimesion
            m_Priors = new double[K];
            m_Means = new double[K][V];
            double[][][] covariances= new double[K][V][V];

            // check if the the initial assignments file exists
            HashMap<Integer,ArrayList<ArrayList<Double>>> topicWiseAssignments= new HashMap<Integer, ArrayList<ArrayList<Double>>>();
            for(int k = 0; k < K; ++ k){
                topicWiseAssignments.put(k,new ArrayList<ArrayList<Double>>());
            }
            for(int d = 0; d < data.length; d ++){
                double[] probVector= new double[K];
                // for each doc create a prob vector with equal probabilities to all topics
                for(int k = 0; k < K; k++)
                    probVector[k]= (double) 1 / K;
                // randomly sample a topic for current doc
                int topic= Tools.sampleFromDistribution(probVector);
                // fill the topicWiseAssignments
                {
                    ArrayList<Double> temp= new ArrayList<Double>();
                    for(double val:data[d])
                        temp.add(val);
                    if(topicWiseAssignments.isEmpty() || topicWiseAssignments.get(topic)==null ){
                        ArrayList<ArrayList<Double>> temp1= new ArrayList<ArrayList<Double>>();
                        temp1.add(temp);
                        topicWiseAssignments.put(topic,temp1);
                    }else{
                        //ArrayList<ArrayList<Double>> temp1= topicWiseAssignments.get(topic);
                        //temp1.add(temp);
                        //topicWiseAssignments.put(topic,temp1);
                        topicWiseAssignments.get(topic).add(temp);
                    }
                }
            }

            // use the topic assignments to calculate the means and variances
            for(int k=0; k< K; k++){
                m_Priors[k]= ((double)topicWiseAssignments.get(k).size()) / (double)data.length;
                double[][] currAssignments= new double[topicWiseAssignments.get(k).size()][V];
                //System.out.println(topicWiseAssignments.get(k));
                for(int n=0; n<topicWiseAssignments.get(k).size(); n++){
                    for(int v=0; v<V; v++){
                        currAssignments[n][v]= topicWiseAssignments.get(k).get(n).get(v);
                    }
                }

                m_Means[k]= Tools.mean(currAssignments, 1);
                covariances[k]= new Covariance(currAssignments).getCovarianceMatrix().getData();

                // check if the covariance is singular if so add some noise
                for(int j=0; j < data[0].length; j++)
                    covariances[k][j][j] += Math.random()/10000;
            }

            ////GMM 聚类
            logger.info("Building GMM Model...");
            MultivariateNormalMixtureExpectationMaximization gmm= new MultivariateNormalMixtureExpectationMaximization(data);
            //gmm.fit(new MixtureMultivariateNormalDistribution(weights, means, covariances),maxIters,1E-5);
            gmm.fit(new MixtureMultivariateNormalDistribution(m_Priors, m_Means, covariances) );
            MixtureMultivariateNormalDistribution gmmModel= gmm.getFittedModel();
            logger.info("Done Building GMM Model");

            //获取Means ,Cov, 和 priors(权重)
            m_Covariances = new double[K][V];
            List<Pair<Double, MultivariateNormalDistribution>> distributions= gmmModel.getComponents();
            int length = distributions.size();
            assert (length == K);
            for(int i = 0; i < K; ++ i){
                m_Priors[i] = distributions.get(i).getFirst();
                double[] t_means =  distributions.get(i).getSecond().getMeans() ;
                RealMatrix rm = distributions.get(i).getSecond().getCovariances();
                assert (t_means.length == V);
                assert (rm.getRowDimension() == V && rm.getColumnDimension() == V);
                for(int j = 0; j < V;++ j){
                    m_Means[i][j] = t_means[j];
                    m_Covariances[i][j] = rm.getEntry(j,j);
                }
            }
        } catch(Exception e){
            logger.error("Exception caugt in GMM model training",e);
            System.exit(1);
        }

    }

    public static double get_gmm_data_posteriors(
            double[][] posteriors, int numClusters, int numData,
            double[] priors, double[][] means, int dimension,
            double [][]covariances, double[][] data){

        int i_d, i_cl; //index
        int dim;
        double LL = 0;

        double halfDimLog2Pi = (dimension / 2.0) * Math.log(2.0 * Math.PI);
        double []logCovariances  = new double[numClusters];
        double [][]invCovariances= new double[numClusters ][ dimension] ;
        double [] logWeights= new double [numClusters] ;

        for (i_cl = 0 ; i_cl < numClusters ; ++ i_cl) {
            double logSigma = 0 ;
            if (priors[i_cl] < 1e-6) {
                logWeights[i_cl] = - Double.MAX_VALUE;
            } else {
                logWeights[i_cl] = Math.log(priors[i_cl]);
            }
            for(dim = 0 ; dim < dimension ; ++ dim) {
                //logSigma += Math.log(covariances[i_cl*dimension + dim]);
                logSigma += Math.log(covariances[i_cl][dim]);
                //invCovariances [i_cl*dimension + dim] =  1.0 / covariances[i_cl*dimension + dim];
                invCovariances [i_cl][dim] =  1.0 / covariances[i_cl][dim];
            }
            logCovariances[i_cl] = logSigma;
        }

        for (i_d = 0 ; i_d < numData ; ++ i_d) {
            double clusterPosteriorsSum = 0;
            //double maxPosterior = (double) (-VL_INFINITY_D) ;
            double maxPosterior = - Double.MAX_VALUE ;

            for (i_cl = 0 ; i_cl < numClusters ; ++ i_cl) {
                double distance = distanceMahalanobis
                        (dimension,data[i_d],means[i_cl],invCovariances[i_cl]);
                double p = logWeights[i_cl]
                                - halfDimLog2Pi
                                - 0.5 * logCovariances[i_cl]
                                - 0.5 * distance;
                //posteriors[i_cl + i_d * numClusters] = p ;
                posteriors[i_d][i_cl] = p ;
                if (p > maxPosterior) { maxPosterior = p ; }
            }

            for (i_cl = 0 ; i_cl < numClusters ; ++ i_cl) {
                //double p = posteriors[i_cl + i_d * numClusters] ;
                double p = posteriors[i_d][i_cl ];
                p =  Math.exp(p - maxPosterior) ;
                //posteriors[i_cl + i_d * numClusters] = p ;
                posteriors[i_d][i_cl] = p ;
                clusterPosteriorsSum += p ;
            }

            LL +=  Math.log(clusterPosteriorsSum) + (double) maxPosterior ;

            for (i_cl = 0 ; i_cl < numClusters ; ++i_cl) {
                //posteriors[i_cl + i_d * numClusters] /= clusterPosteriorsSum ;
                posteriors[i_d][i_cl] /= clusterPosteriorsSum ;
            }
        } /* end of parallel region */


    return  LL;

    }

    private static double distanceMahalanobis(int dimension, double []X, double[] MU, double[] S) {
        if(null == X || null == MU || null == S)
            return Double.MAX_VALUE;
        double acc  = 0;
        double vacc;

        for(int i = 0; i < dimension;++ i){
            double a = X[i], b = MU[i], c = S[i];
            double delta = a - b;
            double delta2 = delta * delta;
            double delta2div = delta2 * c;
            acc += delta2div;
        }
        return acc ;
    }

    public static void main(String[] args) throws Exception {
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
        GMM gmm= new GMM();
        gmm.runEM(data, 5, 100);
        logger.info("训练完毕 ");


        logger.info("保存前 Means");
        Tools.printArray(gmm.getM_Covariances());

        logger.info("-----------------------------------------------------");
        //保存GMM
        SerializationHelper.write("gmm",gmm);

        //加载GMM
        GMM gmm1 = (GMM) SerializationHelper.read("gmm");
        logger.info("保存后 Means");
        Tools.printArray(gmm1.getM_Covariances());


    }

}
