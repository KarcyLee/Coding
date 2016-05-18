package com.sohu.Coding.Tools;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashSet;
import java.util.Set;

/**
 * Created by pengli211286 on 2016/5/18.
 */
public class Distance {
    private static Logger logger = LoggerFactory.getLogger(Distance.class);

    public static double euclideanDistance(double[] p1, double[] p2){
        double sum = 0;
        for (int i = 0; i < p1.length; i++) {
            sum += (( p1[i] - p2[i] ) * ( p1[i] - p2[i] ));
        }
        return Math.sqrt(sum);
    }
    public static double  MinkowskiDistance(double[] p1, double[] p2,double power){
        //欧式距离的母体，欧式距离中power为2
        double sum = 0;
        for (int i = 0; i < p1.length; i++) {
            sum += Math.pow(Math.abs(p1[i] - p2[i]), power);
        }
        return Math.pow(sum, 1 / power);
    }

    public static double ManhattanDistance(double[] p1, double[] p2) {
        double result = 0.0;
        for (int i = 0; i < p1.length; i++) {
            result += Math.abs(p2[i] - p1[i]);
        }
        return result;
    }
    public static double CosineDistance(double[] p1, double[] p2){
        return 1 - CosineSimilarity(p1,p2);
    }
    public static double WeightedCosineDistance(double[] vector1, double[] vector2, double[] weights){
        //1 - (sum(X1.*W*X2)/((sum(X1.*X1))^(1/2)*(sum(X2.*X2))^(1/2)))
        double cosineDist=0;
        double numerator=0, denominator1=0, denominator2=0;
        for(int index=0; index < vector1.length; index++){
            numerator= numerator + vector1[index]*vector2[index]*weights[index];
        }
        for(int index=0; index < vector1.length; index++){
            denominator1= denominator1 + vector1[index]*vector1[index]*weights[index];
        }
        denominator1= Math.pow(denominator1, 0.5);
        for(int index=0; index < vector2.length; index++){
            denominator2= denominator2 + vector2[index]*vector2[index]*weights[index];
        }
        denominator2= Math.pow(denominator2, 0.5);
        cosineDist= 1- (numerator/(denominator1*denominator2));
        if(Double.isNaN(cosineDist))
            cosineDist=1;
        return cosineDist;
    }

    public static double RBFDistance (double[] p1, double[] p2){
        return 1 - RBFSimilarity(p1,p2);
    }
    public static double ChebychevDistance (double[] p1, double[] p2){
        double totalMax = 0.0;
        for (int i = 0; i < p1.length; i++) {
            totalMax = Math.max(totalMax, Math.abs(p1[i] - p2[i]));
        }
        return totalMax;

    }
    public static double JaccardIndexDistance (double[] p1, double[] p2){
        return 1 - JaccardSimilarity(p1,p2);
    }
    public static double SpearmanFootruleDistance(double[] p1, double[] p2){
        long k = p1.length;
        long denom;
        if(k % 2 == 0)
            denom=( k * k )/ 2;
        else
            denom=(( k + 1 ) * ( k - 1))/ 2;
        double sum = 0.0;
        for (int i = 0; i < p1.length; ++ i) {
            double diff = Math.abs(p1[i] - p2[i]);
            sum += diff;
        }
        return 1.0 - (sum / ((double) denom));
    }

    //计算KL散度（相对熵）：当两个随机分布相同时，其相对熵为0.当两个随机分布的差别增加时，器相对熵也增加
    public static double KLDivergence(double[] vector1, double[] vector2){
        //(Kullback-Leibler Divergence）也叫做相对熵（Relative Entropy)
        int numAttributes = vector1.length;
        double klDivergence=0;
        // formula for average KL divergence between two distributions is
        // KLDivergence= sigma_i {p_i*log(p_i/*q_i)}
        double p_i,q_i;
        for(int i = 0; i < numAttributes; ++ i){
            p_i = vector1[i];
            q_i = vector2[i];
            double firstLogTerm = 0.0;

            if(p_i == 0)
                p_i = 1E-10;
            if(q_i == 0)
                q_i = 1E-10;
            firstLogTerm = Math.log(p_i / q_i) / Math.log(2);
            klDivergence = klDivergence+ (p_i * firstLogTerm) ;
        }
        return klDivergence;
    }


    public static double euclideanSimilarity(double[] p1, double[] p2){
        return 1.0 / (1 + euclideanDistance(p1,p2));
    }
    public static double CosineSimilarity (double[] p1, double[] p2){
        double sumTop = 0;
        double sumOne = 0;
        double sumTwo = 0;
        for (int i = 0; i < p1.length; i++) {
            sumTop += p1[i] * p2[i];
            sumOne += p1[i] * p1[i];
            sumTwo += p2[i] * p2[i];
        }
        double denominator = (Math.sqrt(sumOne) * Math.sqrt(sumTwo));
        denominator = Math.max(1e-6,denominator);//防止除0
        double cosSim = sumTop / denominator ;
        if (cosSim < 0)
            cosSim = 0;//This should not happen, but does because of rounding errorsl
        return cosSim;
    }
    public static double PearsonCorrelationCoefficient(double[] p1, double[] p2){
        double xy = 0, x = 0, x2 = 0, y = 0, y2 = 0;
        for (int i = 0; i < p1.length; i++) {
            xy += p1[i] * p2[i];
            x += p1[i];
            y += p2[i];
            x2 += p1[i] * p1[i];
            y2 += p2[i] * p2[i];
        }
        int n = p1.length;
        return (xy - (x * y) / n) / Math.sqrt((x2 - (x * x) / n) * (y2 - (y * y) / n));
    }
    public static double RBFSimilarity (double[] p1, double[] p2){
        double gamma = 0.01;
        if (p1.equals(p2))
            return 1.0;
        double result = Math.exp(gamma * (2.0 * dotProduct(p1, p2) - dotProduct(p1, p1) - dotProduct(p2, p2)));
        return result;
    }
    public static double MaxProductSimilarity (double[] p1, double[] p2){
//        Specialized similarity that takes the maximum product of two feature values.
//        If this value is zero, the similarity is undefined. This similarity measure
//        is used mainly with features extracted from cluster models.
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < p1.length; i++) {
            double v = p1[i] * p2[i];
            if (v > max)
                max = v;
        }
        if (max > 0.0)
            return max;
        else
            return Double.NaN;
    }
    public static double JaccardSimilarity (double[] p1, double[] p2){
//        * Jaccard index. The distance between two sets is computed in the following
//        * way:
//        *               n1 + n2 - 2*n12
//        * D(S1, S2) = ------------------
//        *               n1 + n2 - n12
//        * D(S1,S2) = |S1 ^ S2|
//        *            ---------
//        *            |S1 u S2|
//        * Where n1 and n2 are the numbers of elements in sets S1 and S2, respectively,
//        * and n12 is the number that is in both sets (section).
//                *
        HashSet<Integer> set1 = new HashSet<Integer>();
        HashSet<Integer> set2 = new HashSet<Integer>();

        for (int i = 0; i < p1.length; i++)
            set1.add((int) p1[i]);

        for (int i = 0; i < p2.length; i++)
            set2.add((int) p2[i]);
        Set<Integer> union = new HashSet<Integer>();
        union.addAll(set1);
        union.addAll(set2);

        Set<Integer> intersection = new HashSet<Integer>();
        intersection.addAll(set1);
        intersection.retainAll(set2);

        return ((double)intersection.size()) / ((double)union.size());
    }
    public static double SpearmanRankCorrelation(double[] p1, double[] p2){
//  * Calculates the Spearman rank correlation of two instances. The value on
//  * position 0 of the instance should be the rank of attribute 0. And so on and so forth.
        long k = p1.length;
        long denom = k * (k * k - 1);
        double sum = 0.0;
        for (int i = 0; i < p1.length; i++) {
            double diff = p1[i] - p2[i];
            sum += (diff * diff);
        }
        return 1.0 - (6.0 * (sum / ((double) denom)));
    }


    private static double dotProduct(double[] p1, double[] p2) {
        double result = 0;
        for (int i = 0; i < p1.length; i++) {
            result += p1[i] * p2[i];
        }
        return result;
    }


}
