__kernel void lin_comb(__global int *pointer_k1,__global float *a_g,__global int *pointer_k2,__global float *b_g,__global float *res_g)
{
    int k1 = *pointer_k1;
    int k2 = *pointer_k2;
    unsigned int i = get_global_id(0);
    res_g[i] = k1 * a_g[i] + k2 * b_g[i];
}
