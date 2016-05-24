package com.github.karcylee.Coding.MeanVector;

import com.github.karcylee.Tools.Tools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

/**
 * Created by KarlLee on 2016/5/19.
 */
public class MeanVector implements Serializable, Cloneable{
    private static final long serialVersionUID = -63667113535655272L;
    private static Logger logger = LoggerFactory.getLogger(MeanVector.class);

    public double[] meanVector_encode(double [][]data){
        if(null == data || data.length == 0 || data[0].length == 0){
            logger.error("pls check the input!");
            return null;
        }
        int numData = data.length;
        int dimension = data[0].length;
        double []mean = Tools.mean(data,1);
        return mean;
    }
}
