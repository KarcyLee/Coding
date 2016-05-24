package com.github.karcylee.Coding.VLAD;

/**
 * Created by KarlLee on 2016/5/18.
 *
 * 编码后向量：Φ^(I) = [v1,v2,...vk,....];
 *
 *  VLAD normalization：
 * VLAD supports a number of different normalization strategies.
 * These are optionally applied in this order:
 * (1)Component-wise mass normalization.
 *      Each vector vk is divided by the total mass of features associated to it
 * (2)Square-rooting.
 *      The function sign(z)*sqrt|z| is applied to all scalar components of the VLAD descriptor.
 ***Component-wise l2 normalization.**
 *      The vectors vkvk are divided by their L2Norm
 ***Global l2 normalization.**
 *      The VLAD descriptor Φ^(I) is divided by its norm ∥Φ^(I)∥2∥Φ^(I)∥2.
 *
 *
 */
public class VLADFlag {
    //归一化方式
    public static final int VLAD_FLAG_NORMALIZE_COMPONENTS = (0x1 << 0);//每个vk 做L2归一化
    public static final int VLAD_FLAG_SQUARE_ROOT = (0x1 << 1);
    public static final int VLAD_FLAG_UNNORMALIZED = (0x1 << 2);
    public static final int VLAD_FLAG_NORMALIZE_MASS = (0x1 << 3);
    public static final int VLAD_FLAG_NORMALIZE_GLOBAL = (0x1 << 4);//对Φ^(I) 做归一化
}
