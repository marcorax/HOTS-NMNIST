//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void fb_ts_gen(__global int *ts, __global int *tau_b, 
                        __global int *n_clusters_b, __global int *ev_i_b,
                        __global int *n_events_b,
                        __global int *tcontext, __global int *ts_mask,
                        __global double *partial_sum,
                        __global int *closest, __global int *bevskip)
{                          
    int i_file = (int) get_global_id(0);
    int nfiles = (int) get_global_size(0);
    int lid = (int) get_local_id(1); 
    int lsize = (int) get_local_size(1);
    
    int n_clusters=*n_clusters_b;   
    int tau=*tau_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    


    float ts_value; // default allocation is private, faster than local
    float tmp_ts_value;  
    int lin_idx;
    
    int ts_i;  
    int tssize = n_clusters;
    int loc_idx;
    int n_iter;

       
    
    // Time Surface serial calculation
    // If the local worker size is less than the number of elements in a Time-
    // Surface, then cycle multiple iterations up until all elements are computed
    n_iter = (int) ceil(((float) tssize)/((float) lsize));
    

    ts_value=0;

    lin_idx = idx2d(i_file, nfiles, ev_i, n_events);

    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && bevskip[i_file]==0){//Zeropad events here are actually -1 padded

        for (int i=0; i<n_iter; i++){   
            loc_idx = lid+i*lsize;
            if (loc_idx<tssize){    
                lin_idx = idx2d(i_file, nfiles, 0, n_clusters) + loc_idx;
                            
                if (ts_mask[lin_idx]==1){
                    tmp_ts_value = exp( ((float)(tcontext[lin_idx]-ts_i)) / (float) tau);
                    if (tmp_ts_value>0 && tmp_ts_value<=1){//floatcheck for overflowing
                        ts_value=tmp_ts_value;

                        if (closest[i_file]==loc_idx){
                            loc_idx = idx2d(i_file, nfiles, lid, lsize); 
                            partial_sum[loc_idx] = partial_sum[loc_idx] +
                                                     (double)ts_value;}
                        else{
                            loc_idx = idx2d(i_file, nfiles, lid, lsize); 
                            partial_sum[loc_idx] = partial_sum[loc_idx] -
                                         (double)ts_value/(n_clusters-1);}
                    }                 
                }  
            }           
            ts_value=0;//reset ts_value
        }            
    }
            
}
