package com.github.karcylee.Coding.FisherVector;

/**
 * Created by KarlLee on 2016/5/16.
 *
 *The improved Fisher Vector (IFV) improves the classification performance of
 * the representation by using to ideas:
 * (1)Non-linear additive kernel.
 *      The Hellinger's kernel (or Bhattacharya coefficient) can be used
 *      instead of the linear one at no cost by signed squared rooting.
 *      This is obtained by applying the function |z|sign(z) to each dimension of the vector Φ(I)(编码后向量)
 *      Other additive kernels can also be used at an increased space or time cost.
 *
 * (2)Normalization.
 * Before using the representation in a linear model (e.g. a support vector machine),
 * the vector Φ(I) is further normalized by the L2 norm
 * (note that the standard Fisher vector is normalized by the number of encoded feature vectors).
 * After square-rooting and normalization, the IFV is often used in a linear classifier such as an SVM.
 *
 *
 * Faster computations
 * In practice, several data to cluster assignments qik(系数) are likely to be very small
 * or even negligible. The fast version of the FV sets to zero all but the largest assignment
 * for each input feature。
 *
 */
public class FisherVectorFlag {
    public static final int FISHER_FLAG_SQUARE_ROOT = (0x1 << 0);
    public static final int FISHER_FLAG_NORMALIZED = (0x1 << 1);//L2归一化 后接线性分类器
    public static final int FISHER_FLAG_IMPROVED  = (FISHER_FLAG_NORMALIZED | FISHER_FLAG_SQUARE_ROOT);//IFV
    public static final int FISHER_FLAG_FAST = (0x1 << 2);
    public static final int FISHER_FLAG_UNNORMALIZED = (0x1 << 3); //只做了对numData的归一化
}


