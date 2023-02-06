//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void w_reduce(__global int *ts, __global int *res_x_b, 
                       __global int *res_y_b, __global int *n_pol_b,
                       __global int *n_clusters_b, __global int *ev_i_b,
                       __global int *n_events_b, __global double *weights,
                       __local double *partial_sum,                          
                       __global int *bevskip)
{
    int i_file = (int) get_local_id(0);
    int nfiles = (int) get_local_size(0);
    int ts_index = (int) get_global_id(1); 
    
    int n_clusters=*n_clusters_b;   
    int res_x=*res_x_b;
    int res_y=*res_y_b;
    int n_pol=*n_pol_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    
    
    int lin_idx;  
    
    int ts_i;  
       
    lin_idx = idx2d(i_file, nfiles, ev_i, n_events);
    
    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && bevskip[i_file]==0){//Zeropad events here are actually -1 padded

        lin_idx = idx5d(i_file, nfiles, 0, n_clusters, 0,
                        res_x, 0, res_y, 0, n_pol) + ts_index;
        
                 
        weights[lin_idx] = work_group_reduce_add(weights[lin_idx])/nfiles;

              
        
            
    }
}
