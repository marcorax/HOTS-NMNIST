//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void th_update(__global int *ts, __global int *n_clusters_b,
                        __global int *ev_i_b, __global int *n_events_b,
                        __global float *lrate_b, __global int *closest,
                        __global float *S, __global float *s_gain_b, __global float *dS,
                        __global double *distances,  __global double *th,
                        __global float *tau_th_b,   __global int *bevskip)
{
    int i_file = (int) get_global_id(0);
    int nfiles = (int) get_global_size(0);
    int cluster_i = (int) get_global_id(1); 
    
    int n_clusters=*n_clusters_b;   
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    
    float lrate=*lrate_b;
    float tau_th = *tau_th_b;
    
    float s_gain = *s_gain_b;
    
    int lin_idx;
    
    int ts_i;  
       
    lin_idx = idx2d(i_file, nfiles, ev_i, n_events);
    
    ts_i = ts[lin_idx];    
        
    if (ts_i!=-1 && bevskip[i_file]==0){//Zeropad events here are actually -1 padded
        if (cluster_i<n_clusters){    
            lin_idx = idx2d(i_file, nfiles, cluster_i, n_clusters);
            
            tau_th=tau_th*th[lin_idx];
           
            if(cluster_i==closest[i_file]){                         
                      
                 th[lin_idx] = th[lin_idx] + th[lin_idx]*(
                                (double)lrate*(double)dS[i_file]*(double)exp((distances[lin_idx]-(double)th[lin_idx])/(double)tau_th)+
                                (double)s_gain*(double)lrate*S[i_file]*(double)exp((distances[lin_idx]-(double)th[lin_idx])/(double)tau_th));
            }
            else if ((distances[lin_idx]-th[lin_idx])<0 && dS[i_file]>=0 && S[i_file]>=0){

                  th[lin_idx] = th[lin_idx] - th[lin_idx]*(
                                (double)lrate*(double)dS[i_file]*exp((distances[lin_idx]-(double)th[lin_idx])/(double)tau_th)+
                                (double)s_gain*(double)lrate*(double)S[i_file]*exp((distances[lin_idx]-(double)th[lin_idx])/(double)tau_th));                    
            }    
        }            
    }
}
