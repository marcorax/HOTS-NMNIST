//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void ts_gen(__global int *xs,__global int *ys, __global int *ps,
                     __global int *ts, __global int *res_x_b, 
                     __global int *res_y_b, __global int *tau_b, 
                     __global int *n_pol_b, __global int *n_clusters_b, 
                     __global int *ev_i_b, __global int *n_events_b, 
                     __global int *tcontext, __global int *ts_mask,
                     __global float *TS, __global int *fevskip)
{
    int i_file = (int) get_global_id(0);
    int nfiles = (int) get_global_size(0);
    int ts_index = (int) get_global_id(1); 
    
    int n_clusters=*n_clusters_b;   
    int res_x=*res_x_b;
    int res_y=*res_y_b;
    int n_pol=*n_pol_b;
    int tau=*tau_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    
  
    float ts_value; // default allocation is private, faster than local
    float tmp_ts_value;  
    int lin_idx;
    int xs_i;
    int ys_i;
    int ps_i;
    int ts_i;  
     
    ts_value=0;

    lin_idx = idx2d(i_file, nfiles, ev_i, n_events);
    
    xs_i = xs[lin_idx];
    ys_i = ys[lin_idx];
    ps_i = ps[lin_idx];
    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && fevskip[i_file]==0){//Zeropad events here are actually -1 padded
               
            
        lin_idx = idx4d(i_file, nfiles, 0, res_x, 0, res_y, 0,
                         n_pol) + ts_index;
                    
        if (ts_mask[lin_idx]==1){
            tmp_ts_value = exp(((float)(tcontext[lin_idx]-ts_i)) 
                            / (float) tau);
                            
            if (tmp_ts_value>=0 && tmp_ts_value<=1){//floatcheck for overflowing
                ts_value=tmp_ts_value;                                                                 
            }                 
        }  
             

        lin_idx = idx4d(i_file, nfiles, 
                        0, res_x, 0, res_y, 0, n_pol)
                        + ts_index;
    
        TS[lin_idx] = ts_value; 

            
               
    }
}
