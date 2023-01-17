//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void class_w_update(__global int *ts, __global int *res_x_b, 
                          __global int *res_y_b, __global int *n_pol_b,
                          __global int *n_clusters_b, __global int *ev_i_b,
                          __global int *n_events_b, __global float *weights,                          
                          __global int *batch_labels, __global float *lrate_b,
                          __global float *dweights, __global int *bevskip)
{
    unsigned int i_file = get_global_id(0);
    unsigned int n_iter;
    
    int n_clusters=*n_clusters_b;   
    int res_x=*res_x_b;
    int res_y=*res_y_b;
    int n_pol=*n_pol_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    
    float lrate=*lrate_b;
    
    int lin_idx;
    int tssize = res_x*res_y*n_pol;
    int loc_idx;
    
    __local int ts_i;  

   
    
    // Time Surface serial calculation
    // If the local worker size is less than the number of elements in a Time-
    // Surface, then cycle multiple iterations up until all elements are computed
    n_iter = (int) ceil(((float) tssize)/((float) get_local_size(1)));
    
    lin_idx = idx2d(i_file, (int) get_global_size(0), ev_i, n_events);
    
    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && bevskip[i_file]==0){//Zeropad events here are actually -1 padded

        for (int i=0; i<n_iter; i++){   
            loc_idx = (int)get_local_id(1)+i*(int) get_local_size(1);
            if (loc_idx<tssize){    
                lin_idx = idx5d(i_file, (int) get_global_size(0), batch_labels[i_file], 
                                n_clusters, 0, res_x, 0, res_y, 0, n_pol)
                                + loc_idx;
                            
                weights[lin_idx] = weights[lin_idx] + ((float)lrate)*dweights[lin_idx];
                
                }  
            }
            
    }
}
