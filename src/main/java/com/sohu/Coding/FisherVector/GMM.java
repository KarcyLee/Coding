package com.sohu.Coding.FisherVector;


import com.sohu.Coding.Tools.Tools;
import org.apache.commons.math3.distribution.MixtureMultivariateNormalDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.fitting.MultivariateNormalMixtureExpectationMaximization;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * Created by pengli211286 on 2016/5/16.
 */
////GaussianMixtureModel
public class GMM implements Serializable, Cloneable{
    private static Logger logger = LoggerFactory.getLogger(GMM.class);


    int K; //聚类中心(Topic)数目
    int maxIters;
    double[][] data;
    String mPath;
    String mName;
    MixtureMultivariateNormalDistribution gmmModel;
    //EM gmmModel;

    public GMM(double[][] data, int numClusters, int iters, String modelPath, String modelName ){
        K= numClusters;
        maxIters= iters;
        this.data= data;
        mPath= modelPath;
        mName= modelName;
    }

    public GMM(double[][] data, GMM trainingModel, String modelPath, String modelName){
        this.data= data;
        K= trainingModel.K;
        mPath= modelPath;
        mName= modelName;
        gmmModel= trainingModel.gmmModel;
    }

    public double[] getWeights() {
        List<Pair<Double, MultivariateNormalDistribution>> distributions= gmmModel.getComponents();
        int length = distributions.size();
        assert (length == K);
        double [] weights = new double[K];
        for(int i = 0; i < K; ++ i){
            weights[i] = distributions.get(i).getFirst();
        }
        return weights;
    }

    public List<double[]> getMeans() {
        List<Pair<Double, MultivariateNormalDistribution>> distributions= gmmModel.getComponents();
        int length = distributions.size();
        assert (length == K);
        List<double[]> Means = new ArrayList<double[]>();
        for(int i = 0; i < K; ++ i){
            Means.add( distributions.get(i).getSecond().getMeans() );
        }
        return Means;
    }

    public List<RealMatrix> getCovariance() {
        List<Pair<Double, MultivariateNormalDistribution>> distributions= gmmModel.getComponents();
        int length = distributions.size();
        assert (length == K);
        List<RealMatrix> Covariance = new ArrayList<RealMatrix>() ;
        for(int i = 0; i < K; ++ i){
            Covariance.add( distributions.get(i).getSecond().getCovariances() );
        }
        return Covariance;
    }


    public void runEM(){
        try{
            //数据归一化
            data= Tools.normalizeFeatures(data);
            int V= data[0].length;
            double[] weights= new double[K];
            double[][] means= new double[K][V];
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
                weights[k]= ((double)topicWiseAssignments.get(k).size()) / (double)data.length;
                double[][] currAssignments= new double[topicWiseAssignments.get(k).size()][V];
                //System.out.println(topicWiseAssignments.get(k));
                for(int n=0; n<topicWiseAssignments.get(k).size(); n++){
                    for(int v=0; v<V; v++){
                        currAssignments[n][v]= topicWiseAssignments.get(k).get(n).get(v);
                    }
                }

                means[k]= Tools.mean(currAssignments, 1);
                covariances[k]= new Covariance(currAssignments).getCovarianceMatrix().getData();

                // check if the covariance is singular if so add some noise
                for(int j=0; j<data[0].length; j++)
                    covariances[k][j][j] += Math.random()/10000;
            }

            ////GMM 聚类
            logger.info("Building GMM Model...");
            MultivariateNormalMixtureExpectationMaximization gmm= new MultivariateNormalMixtureExpectationMaximization(data);
            gmm.fit(new MixtureMultivariateNormalDistribution(weights, means, covariances),maxIters,1E-5);
            gmmModel= gmm.getFittedModel();
            logger.info("Done Building GMM Model");


			/*System.out.println("Building GMM Model");

			// train using EM clustering algorithm
			gmmModel = new EM();
			gmmModel.setMaxIterations(maxIters);
			gmmModel.setNumClusters(K);
			gmmModel.buildClusterer(trainingInstances);
			double[][][] clustStatistics= gmmModel.getClusterModelsNumericAtts();
			for(int i=0; i<K; i++){
				System.out.println("Cluster: "+(i+1)+" Mean: ");
				for(int j=0; j<trainingInstances.numAttributes(); j++){
					System.out.print(clustStatistics[i][j][0]+",");
				}
				System.out.println();
				System.out.println("Cluster: "+(i+1)+" StdDev: ");
				for(int j=0; j<trainingInstances.numAttributes(); j++){
					System.out.print(clustStatistics[i][j][1]+",");
				}
				System.out.println();
			}
			// extract mixture coefficients for each of the training instance
			mixtureCoeffs= new double[trainingInstances.numInstances()][K];
			int count=0;
			for(Instance inst: trainingInstances){
				mixtureCoeffs[count]= gmmModel.distributionForInstance(inst);
				//System.out.println((Utilities.max(mixtureCoeffs[count], 1)[0]+1));
				count++;
			}
			*/

        } catch(Exception e){
            logger.error("Exception caugt in GMM model training",e);
            System.exit(1);
        }

    }

	/*public EM getGMMModel(){
		return gmmModel;
	}*/

    public MixtureMultivariateNormalDistribution getGMMModel(){
        return gmmModel;
    }

    public void cleanUpVariables(){
        data= null;
    }

    public static void main(String[] args) throws Exception {
        Random rd = new Random();
        int nRow = 20000;
        int nCol = 200;
        double[][] data= new double[nRow][nCol];
        for(int i = 0; i < nRow;++i ){
            for(int j = 0; j < nCol; ++j ){
                data[i][j] = rd.nextDouble();
                //System.out.println(data[i][j]);
            }
        }
        logger.info("数据完毕 ");
        GMM gmm= new GMM(data, 100, 100,"./model","GMMModel");
        gmm.runEM();

        logger.info("训练完毕 ");
    }

}
