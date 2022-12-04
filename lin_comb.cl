__kernel void lin_comb(float k1, float *a_g, float k2, float *b_g, float *res_g)
{
    unsigned int i = get_global_id(0);
    res_g[i] = k1 * a_g[i] + k2 * b_g[i];
}
