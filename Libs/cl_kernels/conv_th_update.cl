//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void conv_th_update(__global int *ts, __global int *n_clusters_b,
                             __global int *ev_i_b, __global int *n_events_b,
                             __global float *lrate_b, __global int *closest,
                             __global float *S, __global float *dS,
                             __global float *distances,  __global float *th, 
                             __global float *tau_th_b, __global int *bevskip,
                             __global int *prec_m_bf)
{
    unsigned int i_file = get_global_id(0);
    unsigned int n_iter;
    
    int n_clusters=*n_clusters_b;   
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    
    float lrate=*lrate_b;
    float tau_th = *tau_th_b;
    
    int lin_idx;
    int i_cluster;
    
    __local int ts_i;  

   
    
    // threshold serial calculation
    // If the local worker size is less than the number of clusters in the layer,
    // then cycle multiple iterations up until all elements are computed
    n_iter = (int) ceil(((float) n_clusters)/((float) get_local_size(1)));
    
    lin_idx = idx2d(i_file, (int) get_global_size(0), ev_i, n_events);
    
    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && bevskip[i_file]==0){//Zeropad events here are actually -1 padded

        for (int i=0; i<n_iter; i++){   
            i_cluster = (int)get_local_id(1)+i*(int) get_local_size(1);
            if (i_cluster<n_clusters){    
                    lin_idx = idx2d(i_file, (int) get_global_size(0), i_cluster, n_clusters);
                    
                    tau_th=tau_th*th[lin_idx];
                    
                    if(i_cluster==closest[i_file]){                         
                         th[lin_idx] = th[lin_idx] +
//                                        lrate*dS[i_file]*exp((distances[lin_idx]-th[lin_idx])/tau_th);
                                        (0.01f)*lrate*S[i_file]*exp((distances[lin_idx]-th[lin_idx])/tau_th);
                    }
//                     else if ((distances[lin_idx]-th[lin_idx])<0 && dS[i_file]>0 && S[i_file]>0){
                    else if (dS[i_file]>0){

                         th[lin_idx] = th[lin_idx] -
//                                        lrate*dS[i_file]*exp((distances[lin_idx]-th[lin_idx])/tau_th);
                                        (0.01f)*lrate*S[i_file]*exp((distances[lin_idx]-th[lin_idx])/tau_th);                    
                    }
                    
                    
                
                }  
            }
            
    }
}
