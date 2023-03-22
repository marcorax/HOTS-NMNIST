//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void w_update(__global int *ts, __global int *res_x_b, 
                       __global int *res_y_b, __global int *n_pol_b,
                       __global int *n_clusters_b, __global int *ev_i_b,
                       __global int *n_events_b, __global double *weights,                          
                       __global int *closest, __global float *lrate_b,
                       __global float *S, __global float *s_gain_b, __global float *dS,
                       __global double *dweights, __global int *bevskip)
{
    int i_file = (int) get_global_id(0);
    int nfiles = (int) get_global_size(0);
    int ts_index = (int) get_global_id(1); 
    int lsize = (int) get_global_size(1);
    
    int n_clusters=*n_clusters_b;   
    int res_x=*res_x_b;
    int res_y=*res_y_b;
    int n_pol=*n_pol_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    
    float lrate=*lrate_b;
    float s_gain=*s_gain_b;
    int lin_idx;  
    int loc_idx;
    int n_iter;  
    int ts_i;  
    int tssize=res_x*res_y*n_pol;
    
    // Time Surface serial calculation
    // If the local worker size is less than the number of elements in a Time-
    // Surface, then cycle multiple iterations up until all elements are computed
    n_iter = (int) ceil(((float) tssize)/((float) lsize));
    
       
    lin_idx = idx2d(i_file, nfiles, ev_i, n_events);
    
    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && bevskip[i_file]==0){//Zeropad events here are actually -1 padded
        for (int i=0; i<n_iter; i++){   
            loc_idx = ts_index+i*lsize;
            if (loc_idx<tssize){    
                lin_idx = idx5d(i_file, nfiles, closest[i_file], n_clusters, 0,
                                res_x, 0, res_y, 0, n_pol) + loc_idx;
        
                weights[lin_idx] = weights[lin_idx] + 
                                  (double)s_gain*(double)S[i_file]*(double)lrate*dweights[lin_idx] + 
                                  (double)dS[i_file]*(double)lrate*dweights[lin_idx];
           }
       }             
    }
}
