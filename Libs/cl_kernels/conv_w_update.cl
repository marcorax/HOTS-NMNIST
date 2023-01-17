//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void conv_w_update(__global int *ts, __global int *surf_x_b, 
                          __global int *surf_y_b, __global int *n_pol_b,
                          __global int *n_clusters_b, __global int *ev_i_b,
                          __global int *n_events_b, __global double *weights_update,                          
                          __global int *closest, __global float *lrate_b,
                          __global float *S, __global float *dS,
                          __global float *dweights, __global int *bevskip)
{
    unsigned int i_file = get_global_id(0);
    unsigned int n_iter;
    
    int n_clusters=*n_clusters_b;   
    int surf_x=*surf_x_b;
    int surf_y=*surf_y_b;
    int n_pol=*n_pol_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    
    float lrate=*lrate_b;
    
    int lin_idx;
    int tssize = surf_x*surf_y*n_pol;
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
                lin_idx = idx5d(i_file, (int) get_global_size(0), closest[i_file], 
                                n_clusters, 0, surf_x, ~0, surf_y, 0, n_pol)
                                + loc_idx;
                
                if (ev_i==0){
                    weights_update[lin_idx] =  (double)(0.01f)*(double)S[i_file]*(double)lrate*(double)dweights[lin_idx] + 
                                               (double)dS[i_file]*(double)lrate*(double)dweights[lin_idx];
                }  
                else{         
                    weights_update[lin_idx] = weights_update[lin_idx] + 
                                              (double)(0.01f)*(double)S[i_file]*(double)lrate*(double)dweights[lin_idx] + 
                                              (double)dS[i_file]*(double)lrate*(double)dweights[lin_idx];
                }
            }  
        }
            
    }
}
