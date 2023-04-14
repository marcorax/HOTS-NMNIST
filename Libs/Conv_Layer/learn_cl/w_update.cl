//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void w_update(__global int *ts, __global int *win_l_b, 
                       __global int *n_pol_b,
                       __global int *n_clusters_b, __global int *ev_i_b,
                       __global int *n_events_b, __global double *weights,                          
                       __global int *closest, __global float *lrate_b,
                       __global float *S, __global float *s_gain_b, __global float *dS,
                       __global double *distances,  __global double *th,
                       __global double *dweights, __global int *bevskip)
{
    int i_file = (int) get_global_id(0);
    int nfiles = (int) get_global_size(0);
    int ts_index = (int) get_global_id(1); 
    int cluster_i = (int) get_global_id(2); 

    
    int n_clusters=*n_clusters_b;   
    int win_l=*win_l_b;
    int n_pol=*n_pol_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    
    float lrate=*lrate_b;
    float s_gain=*s_gain_b;
    
    
    int lin_idx;
    
    int ts_i;  
       
    lin_idx = idx2d(i_file, nfiles, ev_i, n_events);
    
    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && bevskip[i_file]==0){//Zeropad events here are actually -1 padded

        lin_idx = idx2d(i_file, nfiles, cluster_i, n_clusters);
                        
        if(cluster_i==closest[i_file]){ 

            lin_idx = idx5d(i_file, nfiles, cluster_i, n_clusters, 0,
                            win_l, 0, win_l, 0, n_pol) + ts_index;
            weights[lin_idx] = weights[lin_idx] + 
                              (double)s_gain*(double)S[i_file]*(double)lrate*dweights[lin_idx] + 
                              (double)dS[i_file]*(double)lrate*dweights[lin_idx];
        
        }            
              
//         else if ((distances[lin_idx]-th[lin_idx])<0 && dS[i_file]>=0 && S[i_file]>=0){
//         
//             lin_idx = idx5d(i_file, nfiles, cluster_i, n_clusters, 0,
//                             win_l, 0, win_l, 0, n_pol) + ts_index;
//         
//             weights[lin_idx] = weights[lin_idx] - 
//                               (double)s_gain*(double)S[i_file]*(double)lrate*dweights[lin_idx] - 
//                               (double)dS[i_file]*(double)lrate*dweights[lin_idx];         
// 
//         }
    }
}
