//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void reduce_dist( __global int *ts, __global int *n_clusters_b, 
                           __global int *ev_i_b, __global int *n_events_b, 
                           __global double *partial_sum, 
                           __global double *distances, __global int *fevskip)
{
    int i_file = (int) get_global_id(0);
    int nfiles = (int) get_global_size(0);
    int lid = (int) get_local_id(1); 
    int lsize = (int) get_local_size(1);
    int cluster_i = (int) get_global_id(2);
    
    int n_clusters=*n_clusters_b;   
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    
  

    int lin_idx;
    int loc_idx;
    
    int ts_i;  

    
   
    lin_idx = idx2d(i_file, (int) get_global_size(0), ev_i, n_events);
    
    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && fevskip[i_file]==0){//Zeropad events here are actually -1 padded
                        
        lin_idx = idx2d(i_file, nfiles, cluster_i, n_clusters);                                            
        loc_idx = idx3d(i_file, nfiles, cluster_i, n_clusters, lid, lsize); 
        distances[lin_idx] = work_group_reduce_add(partial_sum[loc_idx]);
        partial_sum[loc_idx]=0; //reset for the next cluster
            
    }
}
