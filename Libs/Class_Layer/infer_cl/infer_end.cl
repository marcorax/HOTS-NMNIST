//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void infer_end(__global int *ts, 
                        __global int *n_clusters_b, __global int *ev_i_b,
                        __global int *n_events_b, __global int *batch_labels,
                        __global double *distances, __global int *closest, 
                        __global int *processed_ev, __global int *correct_ev,
                        __global int *fevskip, __global int *bevskip)
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
                if (distances[lin_idx]<min_distance){
                    closest[i_file]=cluster_i;
                    min_distance=distances[lin_idx];
                }
            }
            else{
                closest[i_file]=cluster_i;
                min_distance=distances[lin_idx];
            }                              
        }    
        processed_ev[i_file] += 1;
    }  
    bevskip[i_file] = fevskip[i_file];     
    if (closest[i_file]==batch_labels[i_file] && ts_i!=-1 && fevskip[i_file]==0){
        correct_ev[i_file] += 1;
    }  
    
    fevskip[i_file]=0;
}
