package com.github.karcylee.Tools;
/**
 * Created by KarlLee on 2016/5/16.
 */

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;

import no.uib.cipr.matrix.DenseCholesky;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;


import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import org.math.array.util.Sorting;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Tools {
    private static Logger logger = LoggerFactory.getLogger(Tools.class);
    /** Constant for normal distribution. */
    public static double m_normConst = Math.log(Math.sqrt(2*Math.PI));

    public static int argmax(double[] x) {
        int i;
        double max = x[0];
        int argmax = 0;
        for (i = 1; i < x.length; i++) {
            if (x[i] > max) {
                max = x[i];
                argmax = i;
            }
        }
        return(argmax);
    }


    ////******************机器学习相关**************************************
    public static int[][] generateConfusionMatrix(int[] actualValues, int[] predictedValues, int numClasses){

        int[][] confMatrix = new int[numClasses][numClasses];
        for(int i=0; i<numClasses; i++)
            for(int j=0; j<numClasses; j++)
                confMatrix[i][j] = 0;
        for(int i=0; i<actualValues.length; i++){
            int rowIndex = actualValues[i]-1;
            int colIndex = predictedValues[i]-1;
            confMatrix[rowIndex][colIndex]++;
        }

        return confMatrix;
    }

    //The following method generates accuracy using the confusion matrix
    public static double calculateAccuracy(int[][] confusionMatrix){
        double accuracy=0;
        double totalDataSize=0;
        for(int index=0; index < confusionMatrix.length; index++)
            for(int index1=0; index1 < confusionMatrix.length; index1++){
                if(index == index1)
                    accuracy= accuracy + confusionMatrix[index][index1];
                totalDataSize= totalDataSize + confusionMatrix[index][index1];;
            }

        return (accuracy/totalDataSize)*100;
    }

     // The following method generates accuracy for a particular using the confusion matrix
    public static double calculateAccuracy(int[][] confusionMatrix, int classIndex){
        double totalDataSize=0;
        for(int index=0; index < confusionMatrix.length; index++){
            totalDataSize= totalDataSize + confusionMatrix[classIndex-1][index];
        }

        return (confusionMatrix[classIndex-1][classIndex-1]/totalDataSize)*100;
    }


     // This method calculates the precision for a particular class using the confusion matrix
    public static double calculatePrecision(int[][] confusionMatrix, int classIndex){
        double precision=0;
        double tp= confusionMatrix[classIndex-1][classIndex-1];
        double fp= 0;
        for(int index=0; index < confusionMatrix.length; index++)
            if(index!=classIndex-1){
                fp= fp + confusionMatrix[index][classIndex-1];
            }
        precision= tp/(tp+fp);
        return precision;
    }

     // This method calculates the recall for a particular class using the confusion matrix
    public static double calculateRecall(int[][] confusionMatrix, int classIndex){
        double precision=0;
        double tp= confusionMatrix[classIndex-1][classIndex-1];
        double fn= 0;
        for(int index=0; index < confusionMatrix.length; index++)
            if(index!=classIndex-1){
                fn= fn + confusionMatrix[classIndex-1][index];
            }
        precision= tp/(tp+fn);
        return precision;
    }

    public static double calculateRMSError(double[] x, double[] y){
        double rms = 0 ;
        for(int i = 0; i < x.length; ++i){
            rms += Math.pow(x[i] - y[i],2);
        }
        rms= rms / x.length;
        rms= Math.sqrt(rms);
        return rms;
    }
    public static double calculateRSquared(double[] expected, double[]  observed) {
        double ssTotal = 0; // total sum of squares
        double expectedMean = mean(expected);
        for(int i=0; i<expected.length; i++){
            ssTotal+= Math.pow(expected[i]-expectedMean,2);
        }
        double ssRes = 0; // sum of squares of residuals
        for(int i=0; i<expected.length; i++){
            ssRes+= Math.pow(expected[i]-observed[i],2);
        }
        return 1 - (ssRes/ssTotal);
    }
    public static double calculateMeanAbsoluteError(double[] expected, double[] observed){
        double error = 0;
        for(int i=0; i<expected.length; i++){
            error += Math.abs(expected[i]-observed[i]);
        }
        return error/expected.length;
    }




//////********* 直方图、区间相关***************************
    //Given a set of ranges, find the bin to which a real number belongs to
    public static int calculateBin(double value, double[] ranges){
        if(ranges.length == 2){
            if(value < ranges[1] && value >= ranges[0])
                return 1;
            else
                return 0;
        }else{
            double[] leftRanges = new double[ranges.length/2+1];
            for(int i=0; i<leftRanges.length; i++){
                leftRanges[i] = ranges[i];
            }
            int leftBin = calculateBin(value, leftRanges);
            if(leftBin > 0){
                return leftBin;
            } else {
                try{
                    double[] rightRanges = new double[(ranges.length-leftRanges.length)+1];
                    for(int i=0; i<rightRanges.length; i++){
                        rightRanges[i] = ranges[ranges.length/2+i];
                    }
                    int rightBin = calculateBin(value, rightRanges);
                    if(rightBin > 0){
                        return ranges.length/2 + rightBin ;
                    } else{
                        return 0;
                    }
                }
                catch(Exception e){
                    e.printStackTrace();
                    return 0;
                }
            }
        }
    }


    ///////////////*************数学相关******************************************
    /* taylor approximation of first derivative of the log gamma function */
    public static double digamma(double x) {
        double p;
        x=x+6;
        p=1/(x*x);
        p=(((0.004166666666667*p-0.003968253986254)*p+
                0.008333333333333)*p-0.083333333333333)*p;
        p=p+Math.log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
        return p;
    }

    ////x>0
    public static double Gamma(double x) {
        // We require x > 0
        // Note that the functions Gamma and LogGamma are mutually dependent.
        // Visit http://www.johndcook.com/stand_alone_code.html for the source of this code and more like it.
		/*if (x <= 0.0)
		{
			std::stringstream os;
	        os << "Invalid input argument " << x <<  ". Argument must be positive.";
	        throw std::invalid_argument( os.str() );
		}*/

        // Split the function domain into three intervals:
        // (0, 0.001), [0.001, 12), and (12, infinity)

        ///////////////////////////////////////////////////////////////////////////
        // First interval: (0, 0.001)
        //
        // For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
        // So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of x^3.
        // The relative error over this interval is less than 6e-7.

        double gamma = 0.577215664901532860606512090; // Euler's gamma constant
        final double DBL_MAX  =1.7976931348623158e+308 ;
        double y;
        int n;
        int arg_was_less_than_one;
        // numerator coefficients for approximation over the interval (1,2)
        final double p[] =
                {
                        -1.71618513886549492533811E+0,
                        2.47656508055759199108314E+1,
                        -3.79804256470945635097577E+2,
                        6.29331155312818442661052E+2,
                        8.66966202790413211295064E+2,
                        -3.14512729688483675254357E+4,
                        -3.61444134186911729807069E+4,
                        6.64561438202405440627855E+4
                };

        // denominator coefficients for approximation over the interval (1,2)
        final double q[] =
                {
                        -3.08402300119738975254353E+1,
                        3.15350626979604161529144E+2,
                        -1.01515636749021914166146E+3,
                        -3.10777167157231109440444E+3,
                        2.25381184209801510330112E+4,
                        4.75584627752788110767815E+3,
                        -1.34659959864969306392456E+5,
                        -1.15132259675553483497211E+5
                };

        double num = 0.0;
        double den = 1.0;
        int i;
        double z ;
        double result;
        double temp;

        if (x < 0.001)
            return 1.0/(x*(1.0 + gamma*x));

        ///////////////////////////////////////////////////////////////////////////
        // Second interval: [0.001, 12)

        if (x < 12.0)
        {
            // The algorithm directly approximates gamma over (1,2) and uses
            // reduction identities to reduce other arguments to this interval.

            y = x;
            n = 0;
            arg_was_less_than_one = (y < 1.0? 1:0);

            // Add or subtract integers as necessary to bring y into (1,2)
            // Will correct for this below
            if (arg_was_less_than_one==1)
            {
                y += 1.0;
            }
            else
            {
                n = (int) (Math.floor(y)) - 1;  // will use n later
                y -= n;
            }

            z = y - 1;
            for (i = 0; i < 8; i++)
            {
                num = (num + p[i])*z;
                den = den*z + q[i];
            }
            result = num/den + 1.0;

            // Apply correction if argument was not initially in (1,2)
            if (arg_was_less_than_one==1)
            {
                // Use identity gamma(z) = gamma(z+1)/z
                // The variable "result" now holds gamma of the original y + 1
                // Thus we use y-1 to get back the orginal y.
                result /= (y-1.0);
            }
            else
            {
                // Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
                for (i = 0; i < n; i++)
                    result *= y++;
            }

            return result;
        }

        ///////////////////////////////////////////////////////////////////////////
        // Third interval: [12, infinity)

        if (x > 171.624)
        {
            // Correct answer too large to display. Force +infinity.
            temp = DBL_MAX;
            return temp*2.0;
        }

        return Math.exp(LogGamma(x));
    }
    public static double LogGamma(double x ) {
        // x must be positive
        // Note that the functions Gamma and LogGamma are mutually dependent.
        // Visit http://www.johndcook.com/stand_alone_code.html for the source of this code and more like it.
		/*if (x <= 0.0)
		{
			std::stringstream os;
	        os << "Invalid input argument " << x <<  ". Argument must be positive.";
	        throw std::invalid_argument( os.str() );
		}*/

        final double c[]=
                {
                        1.0/12.0,
                        -1.0/360.0,
                        1.0/1260.0,
                        -1.0/1680.0,
                        1.0/1188.0,
                        -691.0/360360.0,
                        1.0/156.0,
                        -3617.0/122400.0
                };

        double z;
        double sum;
        double series;
        int i;
        final double halfLogTwoPi = 0.91893853320467274178032973640562;
        double logGamma ;
        if (x < 12.0)
        {
            return Math.log(Math.abs(Gamma(x)));
        }

        // Abramowitz and Stegun 6.1.41
        // Asymptotic series should be good to at least 11 or 12 figures
        // For error analysis, see Whittiker and Watson
        // A Course in Modern Analysis (1927), page 252

        z = 1.0/(x*x);
        sum = c[7];
        for (i=6; i >= 0; i--)
        {
            sum *= z;
            sum += c[i];
        }
        series = sum/x;
        logGamma = (x - 0.5)*Math.log(x) - x + halfLogTwoPi + series;
        return logGamma;
    }
    public static double[] log(double[] vec){
        double[] result= new double[vec.length];
        for(int i=0; i<vec.length; i++)
            result[i]= Math.log(vec[i]);
        return result;
    }

    public static double logmvgamma(double x, double d){
        double y= (d * (d - 1) / 4) * Math.log(Math.PI);
        //System.out.println(y);
        for(int i=0; i<d; i++){
            y+= LogGamma(x-((double)i/2));
            //System.out.println(x-((double)i/2));
        }
        return y;
    }

    //given log(a) and log(b), return log(a + b)
    public static double log_sum(double log_a, double log_b) {
        double v;
        if (log_a < log_b) {
            v = log_b+Math.log(1 + Math.exp(log_a-log_b));
        }
        else {
            v = log_a+Math.log(1 + Math.exp(log_b-log_a));
        }
        return(v);
    }

    //// 获取正太分布的密度值 Density function of normal distribution.
    public static double logNormalDens (double x, double mean, double stdDev) {

        double diff = x - mean;
        return - (diff * diff / (2 * stdDev * stdDev))  - m_normConst - Math.log(stdDev);
        //return Math.log(new NormalDistribution(mean,stdDev).probability(x-(stdDev/100),x+(stdDev/100)));
    }
    ///获取概率估计 Get a probability estimate for a value
    public static double normalPDF(double data, double mean, double stdDev) {

		/*data = Math.rint(data / precision) * precision;
	    double zLower = (data - mean - (precision / 2)) / sigma;
	    double zUpper = (data - mean + (precision / 2)) / sigma;
	    double pLower = Statistics.normalProbability(zLower);
	    double pUpper = Statistics.normalProbability(zUpper);
	    return pUpper - pLower;*/
        return Math.exp(logNormalDens(data, mean, stdDev));
    }

    public static double max(double[] vector){
        double maxVal= -1000000.0;
        for(double val: vector){
            if(val > maxVal)
                maxVal= val;
        }
        return maxVal;
    }
    //Finds the highest N numbers in a vector and returns an array of indices
    public static int[] max(double[] vector, int N){
        double[] newvec= new double[vector.length];
        for(int i=0; i<vector.length;i++)
            newvec[i]= vector[i];
        Sorting s= new Sorting(newvec, false);
        int[] allIndices= s.getIndex();
        int[] nIndices= new int[N];
        int count=0;
        for(int index=vector.length-1;index>=vector.length-N;index--){
            nIndices[count]= allIndices[index];
            count++;
        }
        return nIndices;
    }
    public static double mean(double[] vector){
        double sum= 0;
        for(double val: vector){
            sum+= val;
        }
        return sum/vector.length;
    }
    public static double mean(ArrayList<Double> vector){
        double sum= 0;
        for(double val: vector){
            sum+= val;
        }
        return sum/vector.size();
    }
    public static double min(double[] vector){
        double minVal= 10000000.0;
        for(double val: vector){
            if(minVal > val)
                minVal= val;
        }
        return minVal;
    }
    public static int[] min(double[] vector, int N){
        double[] newvec= new double[vector.length];
        for(int i=0; i<vector.length;i++)
            newvec[i]= vector[i];
        Sorting s= new Sorting(newvec, false);
        int[] allIndices= s.getIndex();
        int[] nIndices= new int[N];
        int count=0;
        for(int index=vector.length-N;index<=vector.length-1;index++){
            nIndices[count]= allIndices[index];
            count++;
        }
        return nIndices;
    }
    public static int[] min(int[] vector, int N){
        double[] newvec= new double[vector.length];
        for(int i=0; i<vector.length;i++)
            newvec[i]= vector[i];
        Sorting s= new Sorting(newvec, false);
        int[] allIndices= s.getIndex();
        int[] nIndices= new int[N];
        int count=0;
        for(int index=vector.length-N;index<=vector.length-1;index++){
            nIndices[count]= allIndices[index];
            count++;
        }
        return nIndices;
    }
    public static double sum(double[] vector){
        double sum= 0;
        for(double val: vector){
            sum+= val;
        }
        return sum;
    }
    public static double[] sum(double[][] vector, int dim){
        double[] sum;
        if(dim==1){
            sum= new double[vector[0].length];
            for(int i=0; i<vector[0].length; i++){
                sum[i]=0;
                for(int j=0; j<vector.length; j++){
                    sum[i]+= vector[j][i];
                }
            }
        }
        else{
            sum= new double[vector.length];
            for(int i=0; i<vector.length; i++){
                sum[i]=0;
                for(int j=0; j<vector[0].length; j++){
                    sum[i]+= vector[i][j];
                }
            }
        }
        return sum;
    }
    public static double variance(ArrayList<Double> vector){
        double var= 0;
        double mean= mean(vector);
        for(double val: vector){
            var+= (val-mean)*(val-mean);
        }
        return var/vector.size();
    }
    public static double weightedMean(double[] vector, double[] weights){
        double numer = 0, denom = 0;
        for(int i=0; i< vector.length; ++i){
            numer += weights[i] * vector[i];
            denom += weights[i];
        }
        return numer/denom;
    }
    // 数组 +=
    public static void arrayAddEqual(double[] A,double[] B){
        if(null == A || null == B){
            logger.error("+= 输入为空！");
            return ;
        }
        if(A.length != B.length){
            logger.error("+= 尺寸不一致！");
            return;
        }
        int len = A.length;
        for(int i = 0; i < len;++ i){
            A[i] += B[i];
        }
    }
    // 数组加法
    public static double[]  arrayAdd(double[] A,double[] B){

        if(null == A || null == B){
            logger.error("+= 输入为空！");
            return null;
        }
        if(A.length != B.length){
            logger.error("+= 尺寸不一致！");
            return null;
        }

        int len = A.length;
        double []result = new double[len];
        for(int i = 0; i < len;++ i){
            result[i] = A[i] + B[i];
        }
        return result;
    }

    /////****************MATLAB 向量、矩阵运算相关******************************

    public static double[][] matrixMultiply(double[][] A, double[][] B, boolean transposeA, boolean transposeB){
		/*Matrix m1= new Matrix(mat1);
		Matrix m2= new Matrix(mat2);
		return m1.times(m2).getArray();*/
        double[][] result= new double[A.length][B[0].length];
        if(transposeA){
            result= new double[A[0].length][B[0].length];
            result= Matrices.getArray(new DenseMatrix(A).transAmult(new DenseMatrix(B),new DenseMatrix(result)));
            //result= new weka.core.matrix.Matrix(A).transpose().times(new weka.core.matrix.Matrix(B)).getArray();
        }
        else
        {
            if(transposeB)
            {	result= new double[A.length][B.length];
                result= Matrices.getArray(new DenseMatrix(A).transBmult(new DenseMatrix(B),new DenseMatrix(result)));
                //result= new weka.core.matrix.Matrix(A).times(new weka.core.matrix.Matrix(B).transpose()).getArray();
            }
            else{
                result= Matrices.getArray(new DenseMatrix(A).mult(new DenseMatrix(B),new DenseMatrix(result)));
                //result= new weka.core.matrix.Matrix(A).times(new weka.core.matrix.Matrix(B)).getArray();
            }
        }
        return result;
    }

    public static double[][] matrixInverse(double[][] mat){

        double[][] inv= new double[mat.length][mat.length];
        DenseMatrix eye= Matrices.identity(mat.length);
        inv= Matrices.getArray(new DenseMatrix(mat).solve(eye, new DenseMatrix(inv)));
        //inv= new weka.core.matrix.Matrix(mat).inverse().getArray();
        return inv;
    }

    public static double[][] matrixTranspose(double[][] mat){
        //return Matrices.getArray(new DenseMatrix(mat).transpose());
        //return new DenseDoubleAlgebra().transpose(new DenseColumnDoubleMatrix2D(mat)).toArray();
        //return new weka.core.matrix.Matrix(mat).transpose().getArray();
        double[][] trans= new double[mat[0].length][mat.length];
        for(int i=0; i<mat.length; i++)
            for(int j=0; j<mat[0].length;j++)
                trans[j][i]= mat[i][j];
        return trans;
    }

     // 计算互相关系数Calculates the cross correlation between two sequences
    public static double calculateCrossCorrelation(double[] x, double[] y){
        double[][] xy= new double[x.length][2];
        for(int j=0; j<x.length; j++){
            xy[j][0]= x[j];
            xy[j][1]= y[j];
        }
        PearsonsCorrelation corrObject= new PearsonsCorrelation(xy);
        return corrObject.getCorrelationMatrix().getData()[0][1];
    }

     ////计算四分位数 Calculates the three quartile values of a given array
    public static double[] calculateQuartiles(double[] vector){
        DescriptiveStatistics stats = new DescriptiveStatistics();
        for( int i = 0; i < vector.length; i++) {
            stats.addValue(vector[i]);
        }
        double quartiles[] = new double[5];
        quartiles[0] = min(vector)-1;
        quartiles[1] = stats.getPercentile(25);
        quartiles[2] = stats.getPercentile(50);
        quartiles[3] = stats.getPercentile(75);
        quartiles[4] = max(vector)+1;
        return quartiles;
    }

    ////Cholesky分解 把矩阵分解为一个下三角矩阵以及它的共轭转置矩阵的乘积
    public static double[][] choleskyDecomposition(double[][] mat) throws Exception{
        DenseCholesky ds= DenseCholesky.factorize(new DenseMatrix(mat));
        double[][] chol=  Matrices.getArray(ds.getU());
        while(containsNegative(getDiagonal(chol))){
            System.out.println("Negative diagonal in Chol");
            for(int i=0; i< mat.length; i++){
                mat[i][i]*=1.2;
            }
            ds= DenseCholesky.factorize(new DenseMatrix(mat));
            chol=  Matrices.getArray(ds.getU());
        }
		/*double[][] chol= new weka.core.matrix.Matrix(mat).chol().getL().transpose().getArray();*/
        return chol;
    }
    //////级联矩阵
    public static double[][] concatArrays(double[][] mat1, double[][] mat2, int direction){
        if(direction == 0) // horizontal concatination
            assert mat1.length == mat2.length;
        else if(direction == 1) // vertical concatination
            assert mat1[0].length == mat2[0].length;
        int rows, cols;
        if(direction == 0){
            rows = mat1.length;
            cols = mat1[0].length + mat2[0].length;
        }
        else{
            rows = mat1.length + mat2.length;
            cols = mat1[0].length;
        }
        double[][] result = new double[rows][cols];
        for(int r = 0; r < mat1.length; r++){
            for(int c = 0; c < mat1[0].length; c++){
                result[r][c] = mat1[r][c];
            }
        }
        for(int r = 0; r < mat2.length; r++){
            for(int c = 0; c < mat2[0].length; c++){
                if(direction == 0)
                    result[r][mat1[0].length+c] = mat2[r][c];
                else
                    result[mat1.length+r][c] = mat2[r][c];
            }
        }

        return result;
    }
    public static boolean containsInfinity(double[][] mat){
        boolean contains= false;
        for(int i=0; i<mat.length; i++){
            for(int j=0; j<mat[0].length; j++){
                if(Double.isInfinite(mat[i][j])){
                    contains= true;
                    break;
                }
            }
        }
        return contains;
    }

    public static boolean containsInfinity(double[]mat){
        boolean contains= false;
        for(int i=0; i<mat.length; i++){
            if(Double.isInfinite(mat[i])){
                contains= true;
                break;
            }
        }
        return contains;
    }

    public static boolean containsNaN(double[][]  mat){
        boolean contains= false;
        for(int i=0; i<mat.length; i++){
            for(int j=0; j<mat[0].length; j++){
                if(Double.isNaN(mat[i][j])){
                    contains= true;
                    break;
                }
            }
        }
        return contains;
    }

    public static boolean containsNaN(double[] vector){
        boolean contains=false;
        for(double num: vector){
            if(Double.isNaN(num)){
                contains= true;
                break;
            }
        }
        return contains;
    }

    public static boolean containsZero(double[] vector){
        boolean contains=false;
        for(double num: vector){
            if(num==0.0){
                contains= true;
                break;
            }
        }
        return contains;
    }

    public static boolean containsNegative(double[] vec){
        boolean contains=false;
        for(double num: vec){
            if(num<0.0){
                contains= true;
                break;
            }
        }
        return contains;
    }

    public static double[] getDiagonal(double[][] mat){
        double[] result= new double[mat.length];
        for(int i=0; i<mat.length; i++)
            result[i]= mat[i][i];
        return result;
    }

    ///// 返回 m x m identity matrix
    public static double[][] eye(int m){
        double[][] identityMat= new double[m][m];
        for(int i=0; i<m; i++)
            for(int j=0; j<m; j++){
                if(i==j)
                    identityMat[i][j]=1;
                else
                    identityMat[i][j]=0;
            }
        return identityMat;
    }

    public static double[][] sqrt(double[][] mat){

        double[][] result= new double[mat.length][mat[0].length];
        for(int i=0; i<mat.length; i++)
            for(int j=0; j<mat[0].length; j++)
                result[i][j]= Math.sqrt(mat[i][j]);
        return result;
    }
    public static double[] mean(double[][] vector, int dim){
        double[] mean;
        if(dim == 1){
            mean = new double[vector[0].length];
            for(int i=0; i<vector[0].length; i++){
                double sum = 0;
                for(int j = 0; j < vector.length; j++){
                    sum += vector[j][i];
                }
                mean[i]= sum / vector.length;
            }
        } else{
            mean= new double[vector.length];
            for(int i=0; i<vector.length; i++){
                double sum = 0;
                for(int j=0; j<vector[0].length; j++){
                    sum+= vector[i][j];
                }
                mean[i]= sum/vector[0].length;
            }
        }
        return mean;
    }
    public static void sizeOf(double[][] mat){
        System.out.println(mat.length+"x"+mat[0].length);
    }
    public static double[] stddeviation(double[][] v) {
        double[] var = variance(v);
        for (int i = 0; i < var.length; i++)
            var[i] = Math.sqrt(var[i]);
        return var;
    }
    public static double[] variance(double[][] v) {
        int m = v.length;
        int n = v[0].length;
        double[] var = new double[n];
        int degrees = (m - 1);
        double c;
        double s;
        for (int j = 0; j < n; j++) {
            c = 0;
            s = 0;
            for (int k = 0; k < m; k++)
                s += v[k][j];
            s = s / m;
            for (int k = 0; k < m; k++)
                c += (v[k][j] - s) * (v[k][j] - s);
            var[j] = c / degrees;
        }
        return var;
    }




    ///////*****************字符串相关**********************************
    /////首字母变大写。
    public static String capitalizeFirstLetter(String word){
        String capitalizedWord= word;
        return new String(""+capitalizedWord.charAt(0)).toUpperCase()+capitalizedWord.substring(1);
    }
    ///生成随机数组
    public static String generateRandomString(int length){
        String[] randomChars= {"A","B","C","D","E","F","G","H","I","J","K","L","M",
                "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
                "1","2","3","4","5","6","7","8","9","0",
                "!","@","#","$","&","%","@","#"};
        String randomString= "";
        Random rnd= new Random();
        for(int i=0; i<length; i++){
            randomString+= new String(""+randomChars[rnd.nextInt(randomChars.length)]);
        }
        return randomString;
    }




///////////////***************打印工具*****************************
    public static void printArray(double[][] array){
        for(int i=0; i<array.length; i++){
            for(int j=0; j<array[i].length; j++)
                System.out.print(String.format("%.3f", array[i][j])+"\t");
            System.out.print("\n");
        }
    }
    public static void printArray(int[][] array){
        for(int i=0; i<array.length; i++){
            for(int j=0; j<array[i].length; j++)
                System.out.print(String.format("%d", array[i][j])+"\t");
            System.out.print("\n");
        }
    }
    public static void printArrayToFile(double[][] array,String filePath) {
        try{
            PrintWriter pw= new PrintWriter(new File(filePath));
            for(int i=0; i<array.length; i++){
                for(int j=0; j<array[i].length; j++)
                    pw.print(String.format("%.4f", array[i][j])+"\t");
                pw.print("\n");
            }
            pw.close();
        }catch(IOException ioe){}

    }
    public static void printArray(int[] array){
        System.out.print("\n[ ");
        for(int i=0; i<array.length; i++){
            System.out.print(array[i]+" ");
        }
        System.out.print("]\n");
    }
    public static void printArray(double[] array){
        System.out.print("\n[ ");
        for(int i=0; i<array.length; i++){
            System.out.print(String.format("%.3f", array[i])+"\t");
        }
        System.out.print("]\n");
    }
    public static void printArray(String message,double[] array){
        DecimalFormat fmt= new DecimalFormat("#.####");
        System.out.print(message+": [ ");
        for(int i=0; i<array.length; i++){
            System.out.print(fmt.format(array[i])+" ");
        }
        System.out.print("]\n");
    }
    public static void printArray(String[] array){
        System.out.print("\n[ ");
        for(int i=0; i<array.length; i++){
            System.out.print(array[i]+" ");
        }
        System.out.print("]\n");
    }



    ////*************************数据处理部分**************************


     //// This method takes a multinomial (or bernoulli) probability distribution as an input and samples
    public static int sampleFromDistribution(double[] p){
        int sample;
        // cumulate multinomial parameters
        for (int k = 1; k < p.length; k++) {
            p[k] += p[k - 1];
        }
        // scaled sample because of unnormalised p[]
        double u = Math.random() * p[p.length-1];
        for (sample = 0; sample < p.length; sample++) {
            if (u < p[sample])
                break;
        }
        return sample;
    }









    //L2归一化,
    public static void NormalizationL2_Replace(double[]A){
        if(null == A || A.length == 0) {
            logger.error("归一化前向量为空，请检查输入");
            return;
        }
        double n = 0 ;
        for(int dim = 0; dim < A.length; ++dim) {
            double z = A[dim] ;
            n += z * z ;
        }
        n = Math.sqrt(n) ;
        n = Math.max(n, 1e-12) ;
        for(int dim = 0; dim < A.length; ++dim) {
            A[dim] /= n ;
        }
    }
    public static double[] NormalizationL2(double[]A){
        if(null == A || A.length == 0) {
            logger.error("归一化前向量为空，请检查输入");
            return null;
        }
        double[] res = new double[A.length];
        double n = 0 ;
        for(int dim = 0; dim < A.length; ++dim) {
            double z = A[dim] ;
            n += z * z ;
        }
        n = Math.sqrt(n) ;
        n = Math.max(n, 1e-12) ;
        for(int dim = 0; dim < A.length; ++dim) {
            res[dim] = A[dim] / n ;
        }
        return res;
    }
    public static void NormalizationL2_Replace(double[][]A){
        if(null == A || A.length == 0 || A[0].length == 0) {
            logger.error("归一化前向量为空，请检查输入");
            return;
        }
        int dimension = A[0].length;
        for(int i_d = 0; i_d < A.length; ++i_d){
            if(A[i_d].length != dimension){
                logger.error("维度不一致！");
                return;
            }
            double n = 0 ;
            for(int dim = 0; dim < dimension; ++dim) {
                double z = A[i_d][dim] ;
                n += z * z ;
            }
            n = Math.sqrt(n) ;
            n = Math.max(n, 1e-12) ;
            for(int dim = 0; dim < dimension; ++dim) {
                A[i_d][dim] /= n ;
            }
        }


    }
    public static double[][] NormalizationL2(double[][]A){
        if(null == A || A.length == 0 || A[0].length == 0) {
            logger.error("归一化前向量为空，请检查输入");
            return null;
        }
        int dimension = A[0].length;
        double [][] res = new double[A.length][dimension];
        for(int i_d = 0; i_d < A.length; ++i_d){
            if(A[i_d].length != dimension){
                logger.error("维度不一致！");
                return null;
            }
            double n = 0 ;
            for(int dim = 0; dim < dimension; ++dim) {
                double z = A[i_d][dim] ;
                n += z * z ;
            }
            n = Math.sqrt(n) ;
            n = Math.max(n, 1e-12) ;
            for(int dim = 0; dim < dimension; ++dim) {
                res[i_d][dim] = A[i_d][dim] / n ;
            }
        }
        return  res;
    }

    //L1归一化,
    public static void NormalizationL1_Replace(double[]A){
        if(null == A || A.length == 0) {
            logger.error("归一化前向量为空，请检查输入");
            return;
        }
        double n = 0 ;
        for(int dim = 0; dim < A.length; ++dim) {
            n += Math.abs(A[dim]) ;
        }
        n = Math.max(n, 1e-12) ;
        for(int dim = 0; dim < A.length; ++dim) {
            A[dim] /= n ;
        }
    }
    public static double[] NormalizationL1(double[]A){
        if(null == A || A.length == 0) {
            logger.error("归一化前向量为空，请检查输入");
            return null;
        }
        double[] res = new double[A.length];
        double n = 0 ;
        for(int dim = 0; dim < A.length; ++ dim) {
            n += Math.abs(A[dim]) ;
        }
        n = Math.max(n, 1e-12) ;
        for(int dim = 0; dim < A.length; ++ dim) {
            res[dim] = A[dim] / n ;
        }
        return res;
    }
    public static void NormalizationL1_Replace(double[][]A){
        if(null == A || A.length == 0 || A[0].length == 0) {
            logger.error("归一化前向量为空，请检查输入");
            return;
        }
        int dimension = A[0].length;
        for(int i_d = 0; i_d < A.length; ++i_d){
            if(A[i_d].length != dimension){
                logger.error("维度不一致！");
                return;
            }
            double n = 0 ;
            for(int dim = 0; dim < dimension; ++dim) {
                n += Math.abs(A[i_d][dim]) ;
            }
            n = Math.max(n, 1e-12) ;
            for(int dim = 0; dim < dimension; ++dim) {
                A[i_d][dim] /= n ;
            }
        }


    }
    public static double[][] NormalizationL1(double[][]A){
        if(null == A || A.length == 0 || A[0].length == 0) {
            logger.error("归一化前向量为空，请检查输入");
            return null;
        }
        int dimension = A[0].length;
        double [][] res = new double[A.length][dimension];
        for(int i_d = 0; i_d < A.length; ++i_d){
            if(A[i_d].length != dimension){
                logger.error("维度不一致！");
                return null;
            }
            double n = 0 ;
            for(int dim = 0; dim < dimension; ++dim) {
                n += Math.abs(A[i_d][dim] );
            }
            n = Math.max(n, 1e-12) ;
            for(int dim = 0; dim < dimension; ++dim) {
                res[i_d][dim] = A[i_d][dim] / n ;
            }
        }
        return  res;
    }

    //归一化到矩阵中的最小值与最大值之间
    public static double[][] normalizeFeatures(double[][] mat){
        double[][] newMat= new double[mat.length][mat[0].length];
        double[] minVals= new double[mat[0].length];
        double[] maxVals= new double[mat[0].length];
        for(int n=0; n<mat[0].length; n++){
            minVals[n]=Double.POSITIVE_INFINITY;
            maxVals[n]=Double.NEGATIVE_INFINITY;
        }
        for(int i=0; i<mat.length; i++){
            for(int n=0; n<mat[0].length; n++){
                if(mat[i][n]<minVals[n]){
                    minVals[n]= mat[i][n];
                }
                if(mat[i][n]>maxVals[n]){
                    maxVals[n]= mat[i][n];
                }
            }
        }

        for(int i=0; i<mat.length; i++){
            for(int n=0; n<mat[0].length; n++){
                if(maxVals[n] == minVals[n])
                    newMat[i][n]= 1;
                else
                    newMat[i][n] = (mat[i][n] - minVals[n]) /
                            (maxVals[n] - minVals[n]);
            }
        }
		/*
		 value = (vals[j] - m_MinArray[j]) /
	      (m_MaxArray[j] - m_MinArray[j]) * m_Scale + m_Translation;
		 */
        return newMat;
    }
    ///将数据限定在min/max之间并归一化Scales an array of real values to the range [min,max]
    public static double[] scaleData(double[] input,double min, double max){
        // find the max and min values of the array
        double originalMin = 10000,originalMax= 0;
        double[] output = new double[input.length];
        for(int i=0; i < input.length; i++){
            if(input[i] < originalMin){
                originalMin= input[i];
            }
            if(input[i] > originalMax){
                originalMax= input[i];
            }
            //System.out.println(input[i]);
        }
        for(int i=0; i<input.length; i++){
            output[i]=(((max-min)/(originalMax-originalMin+0.00001))*(input[i]-originalMin))+min;
        }
        return output;
    }
    public static double[] scaleData(double[] input,double originalMin, double originalMax, double targetMin, double targetMax){
        double[] output= new double[input.length];
        //System.out.println(min+","+max+","+originalMin+","+originalMax);
        for(int i=0; i<input.length; i++){
            output[i]=(((targetMax-targetMin)/(originalMax-originalMin+0.00001))*(input[i]-originalMin))+targetMin;
            //System.out.println(input[i]+","+output[i]);
        }
        return output;
    }



}

