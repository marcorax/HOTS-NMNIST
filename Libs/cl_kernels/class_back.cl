//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b
//TODO IMPLEMENT TS_DROP Boolean for when rec_distances_i[rec_closest_i]-th_i[rec_closest_0])<0
__kernel void class_back(__global int *xs,__global int *ys, __global int *ts,
                          __global int *res_x_b, __global int *res_y_b,
                          __global int *tau_b, __global int *n_pol_b,
                          __global int *n_clusters_b, __global int *ev_i_b,
                          __global int *n_events_b, __global int *tcontext_fb,
                          __global int *ts_mask_fb, __global float *weights,
                          __local float *partial_sum, __global float *S,
                          __global int *closest, __global float *TS)
{
    unsigned int i_file = get_global_id(0);
    unsigned int n_iter_fb;
    unsigned int n_iter;

    
    int n_clusters=*n_clusters_b;   
    int res_x=*res_x_b;
    int res_y=*res_y_b;
    int n_pol=*n_pol_b;
    int tau=*tau_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;        

    float ts_value; // default allocation is private, faster than local
    float tmp_ts_value;  
    int lin_idx;
    __local int xs_i;
    __local int ys_i;
    __local int ps_i;
    __local int ts_i;  
    float elem_distance; 
    float min_distance;
    int tssize = res_x*res_y*n_pol;
    int tsidx;
    
   
    
    // Time Surface serial calculation
    // If the local worker size is less than the number of elements in a Time-
    // Surface, then cycle multiple iterations up until all elements are computed
    n_iter = (int) ceil(((float) tssize)/((float) get_local_size(1)));
    

    ts_value=0;

    lin_idx = idx2d(i_file, (int) get_global_size(0), ev_i, n_events);
    
    
    xs_i = xs[lin_idx];
    ys_i = ys[lin_idx];
    ps_i = *prev_closest;
    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1){//Zeropad events here are actually -1 padded
        lin_idx = idx4d(i_file, (int) get_global_size(0), xs_i, res_x, ys_i, res_y,
                        ps_i, n_pol);
                        
        tcontext[lin_idx] = ts_i;
        
        if (ts_mask[lin_idx]==0){
            ts_mask[lin_idx]=1;}
            
        for (int i=0; i<n_iter; i++){   
            tsidx = (int)get_local_id(1)+i*(int) get_local_size(1);
            if (tsidx<tssize){    
                lin_idx = idx4d(i_file, (int) get_global_size(0), 0, res_x, 
                                0, res_y, 0, n_pol) + tsidx;
                            
                if (ts_mask[lin_idx]==1){
                    tmp_ts_value = exp( ((float)(tcontext[lin_idx]-ts_i)) / (float) tau);
                    if (tmp_ts_value>=0 && tmp_ts_value<=1){//doublecheck for overflowing
                        ts_value=tmp_ts_value;                                                                 
                    }                 
                }  
            }
        lin_idx = idx5d(i_file, (int) get_global_size(0), 
                        ev_i, n_events, 0, res_x, 0, res_y, 0, n_pol)
                        + (int) get_local_id(1) 
                        + i* (int) get_local_size(1);
    
        TS[lin_idx] = ts_value; //Leftover for debug 
        ts_value=0;//reset ts_value

        }    
    }
        

    for (int cl=0; cl<n_clusters; cl++){
        for (int i=0; i<n_iter; i++){   
            tsidx = (int)get_local_id(1)+i*(int) get_local_size(1);
            if (tsidx<tssize){  
            
                lin_idx = idx5d(i_file, (int) get_global_size(0), 
                                ev_i, n_events, 0, res_x, 0, res_y, 0, n_pol)
                                + (int) get_local_id(1) 
                                + i* (int) get_local_size(1);
            
                ts_value=TS[lin_idx]; //Leftover for debug  
            
            
                lin_idx = idx5d(i_file, (int) get_global_size(0), cl, 
                                n_clusters, 0, res_x, 0, res_y, 0, n_pol)
                                + (int) get_local_id(1) 
                                + i* (int) get_local_size(1);
            
                //Euclidean is causing a good chunk of approx errors, moving to L1 
                elem_distance = fabs(weights[lin_idx]-ts_value);
                partial_sum[(int) get_local_id(1)]+=elem_distance;
            }
        } 
        
                
        //REDUCTION ALGORITHM HERE    
        lin_idx = idx3d(i_file, (int) get_global_size(0), ev_i, n_events, cl,
                          n_clusters);                                            
                          
        distances[lin_idx] = work_group_reduce_add(partial_sum[(int) get_local_id(1)]);
        partial_sum[(int) get_local_id(1)]=0; //reset for the next cluster

        if (get_local_id(1)==0){
        
            if(cl!=0){
            
                if (distances[lin_idx]<min_distance){
                    closest[i_file]=cl;
                    min_distance=distances[lin_idx];
                }

            }
            else{
                closest[i_file]=cl;
                min_distance=distances[lin_idx];
            }
        }
                
    }

    if(i_file==0 && get_local_id(1)==0){
        *ev_i_b=ev_i+1;
    }    
    
}
