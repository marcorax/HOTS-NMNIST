//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void infer_end(__global int *xs,__global int *ys,
                        __global int *ps, __global int *ts, 
                        __global int *n_clusters_b, __global int *ev_i_b,
                        __global int *n_events_b, __global double *th,
                        __global double *distances, __global int *closest, 
                        __global int *fevskip)
{
    int i_file = (int) get_global_id(0);
    int nfiles = (int) get_global_size(0);
    
    int n_clusters=*n_clusters_b;   
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    
    
    int lin_idx;
    int ts_i;  
    double min_distance;
    
   
    
    lin_idx = idx2d(i_file, (int) get_global_size(0), ev_i, n_events);
    
    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && fevskip[i_file]==0){//Zeropad events here are actually -1 padded        
        for (int cluster_i=0; cluster_i<n_clusters; cluster_i++){  
            lin_idx = idx2d(i_file, nfiles, cluster_i, n_clusters);             
            if(cluster_i!=0){         
                if (distances[lin_idx]-th[lin_idx]<min_distance){
                    closest[i_file]=cluster_i;
                    min_distance=distances[lin_idx]-th[lin_idx];
                }
            }
            else{
                closest[i_file]=cluster_i;
                min_distance=distances[lin_idx]-th[lin_idx];
            }                              
        }    
        
        lin_idx = idx2d(i_file, nfiles, ev_i, n_events);
        ps[lin_idx] = closest[i_file];
        xs[lin_idx] = 0;
        ys[lin_idx] = 0;

    
        lin_idx = idx2d(i_file, nfiles, closest[i_file], n_clusters);
        if (min_distance>th[lin_idx]){
            fevskip[i_file] = 1;
        }
        
        
  
    }
}
