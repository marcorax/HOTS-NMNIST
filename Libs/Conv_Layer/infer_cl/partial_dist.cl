//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void partial_dist(__global int *ts, __global int *win_l_b, 
                           __global int *n_pol_b,
                           __global int *n_clusters_b, __global int *ev_i_b,
                           __global int *n_events_b, __global double *weights,
                           __global double *partial_sum, __global float *TS,
                           __global double *dweights, __global int *fevskip)
{
    int i_file = (int) get_global_id(0);
    int nfiles = (int) get_global_size(0);
    int lid = (int) get_local_id(1); 
    int lsize = (int) get_local_size(1);
    int cluster_i = (int) get_global_id(2);

    int n_iter;
    
    int n_clusters=*n_clusters_b;   
    int win_l=*win_l_b;
    int n_pol=*n_pol_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    
  

    double ts_value; 
    int lin_idx;
    int ts_i;  
    double elem_distance; 
    int tssize = win_l*win_l*n_pol;
    int loc_idx;
    
   
    
    // Time Surface serial calculation
    // If the local worker size is less than the number of elements in a Time-
    // Surface, then cycle multiple iterations up until all elements are computed
    n_iter = (int) ceil(((float) tssize) / ((float) lsize));
    

    ts_value=0;

    lin_idx = idx2d(i_file, nfiles, ev_i, n_events);
    
    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && fevskip[i_file]==0){//Zeropad events here are actually -1 padded
                      
        for (int i=0; i<n_iter; i++){   
            loc_idx = lid+i*lsize;
            if (loc_idx<tssize){  

                                
                lin_idx = idx4d(i_file, nfiles, 
                                0, win_l, 0, win_l, 0, n_pol) +
                                loc_idx;
                        
                ts_value= (double) TS[lin_idx]; 
            
            
                lin_idx = idx5d(i_file, nfiles, cluster_i, 
                                n_clusters, 0, win_l, 0, win_l, 0, n_pol) +
                                loc_idx;
            
                //Euclidean is causing a good chunk of approx errors, moving to L1 
//                     elem_distance = fabs(weights[lin_idx]-ts_value);
                elem_distance = (weights[lin_idx]-ts_value)*
                                (weights[lin_idx]-ts_value);


                //save the weight change for the fb. to save computation
                dweights[lin_idx] = ts_value-weights[lin_idx];
                loc_idx = idx3d(i_file, nfiles, cluster_i, n_clusters, lid,
                               lsize); 
                partial_sum[loc_idx] = partial_sum[loc_idx] + elem_distance;
                
            }
        }               
    }
}
