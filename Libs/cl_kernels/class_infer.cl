//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void class_infer(__global int *xs,__global int *ys, __global int *ts,
                          __global int *res_x_b, __global int *res_y_b,
                          __global int *tau_b, __global int *n_pol_b,
                          __global int *n_clusters_b, __global int *ev_i_b,
                          __global int *n_events_b, __global int *tcontext,
                          __global int *ts_mask, __global float *weights,
                          __global float *partial_sum, __global float *distances,
                          __global int *closest, __global int *prev_closest,
                          __global float *TS, __global float *dweights, 
                          __global int *fevskip, __global int *bevskip, 
                          __global int *processed_ev, __global int *correct_ev, 
                          __global int *batch_labels)
{
    unsigned int i_file = get_global_id(0);
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
    int loc_idx;
    
   
    
    // Time Surface serial calculation
    // If the local worker size is less than the number of elements in a Time-
    // Surface, then cycle multiple iterations up until all elements are computed
    n_iter = (int) ceil(((float) tssize) / ((float) get_local_size(1)));
    

    ts_value=0;

    lin_idx = idx2d(i_file, (int) get_global_size(0), ev_i, n_events);
    
    
    xs_i = xs[lin_idx];
    ys_i = ys[lin_idx];
    ps_i = *prev_closest;
    ts_i = ts[lin_idx];   
        
    if (xs_i!=-1 && fevskip[i_file]==0){//Zeropad events here are actually -1 padded

        lin_idx = idx4d(i_file, (int) get_global_size(0), xs_i, res_x, ys_i, res_y,
                        ps_i, n_pol);
                        
        tcontext[lin_idx] = ts_i;
        
        if (ts_mask[lin_idx]==0){
            ts_mask[lin_idx]=1;}
            
        for (int i=0; i<n_iter; i++){   
            loc_idx = (int)get_local_id(1)+i*(int) get_local_size(1);
            if (loc_idx<tssize){    
                lin_idx = idx4d(i_file, (int) get_global_size(0), 0, res_x, 
                                0, res_y, 0, n_pol) + loc_idx;
                            
                if (ts_mask[lin_idx]==1){
                    tmp_ts_value = exp( ((float)(tcontext[lin_idx]-ts_i)) / (float) tau);
                    if (tmp_ts_value>=0 && tmp_ts_value<=1){//doublecheck for overflowing
                        ts_value=tmp_ts_value;                                                                 
                    }                 
                }  
            
            
                //Debug Index (save each event)
        //         lin_idx = idx5d(i_file, (int) get_global_size(0), 
        //                         ev_i, n_events, 0, res_x, 0, res_y, 0, n_pol)
        //                         + (int) get_local_id(1) 
        //                         + i* (int) get_local_size(1);
        
                lin_idx = idx4d(i_file, (int) get_global_size(0), 
                                0, res_x, 0, res_y, 0, n_pol)
                                + (int) get_local_id(1) 
                                + i* (int) get_local_size(1);
            
                TS[lin_idx] = ts_value; 
                ts_value=0;//reset ts_value
    
            }
        }
        
        if (get_local_id(1)==0){
        
            processed_ev[i_file] += 1;       
        
        }    
    
        
        for (int cl=0; cl<n_clusters; cl++){
            for (int i=0; i<n_iter; i++){   
                loc_idx = (int)get_local_id(1)+i*(int) get_local_size(1);
                if (loc_idx<tssize){  
                    //Debug Index (save each event)
    //                 lin_idx = idx5d(i_file, (int) get_global_size(0), 
    //                                 ev_i, n_events, 0, res_x, 0, res_y, 0, n_pol)
    //                                 + (int) get_local_id(1) 
    //                                 + i* (int) get_local_size(1);
                                    
                    lin_idx = idx4d(i_file, (int) get_global_size(0), 
                                    0, res_x, 0, res_y, 0, n_pol) +
                                    loc_idx;
                            
                    ts_value=TS[lin_idx]; 
                
                
                    lin_idx = idx5d(i_file, (int) get_global_size(0), cl, 
                                    n_clusters, 0, res_x, 0, res_y, 0, n_pol) +
                                    loc_idx;
                
                    //Euclidean is causing a good chunk of approx errors, moving to L1 
//                     elem_distance = fabs(weights[lin_idx]-ts_value);
                    elem_distance = pow(weights[lin_idx]-ts_value, 2.0f);

                    //save the weight change for the fb. to save computation
                    dweights[lin_idx] = ts_value-weights[lin_idx];
                    loc_idx = idx2d(i_file, (int) get_global_size(0), (int) get_local_id(1),
                                   (int) get_local_size(1)); 
                    partial_sum[loc_idx] = partial_sum[loc_idx] + elem_distance;
                }
            } 
            
                    
            //REDUCTION ALGORITHM HERE    
            lin_idx = idx2d(i_file, (int) get_global_size(0), cl, n_clusters);                                            
            loc_idx = idx2d(i_file, (int) get_global_size(0), (int) get_local_id(1),
                           (int) get_local_size(1)); 
            distances[lin_idx] = work_group_reduce_add(partial_sum[loc_idx]);
            partial_sum[loc_idx]=0; //reset for the next cluster
    
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
    
    }
    
    if (get_local_id(1)==0){
    
        bevskip[i_file] = fevskip[i_file];     
        if (closest[i_file]==batch_labels[i_file] && ts_i!=-1 && fevskip[i_file]==0){
            correct_ev[i_file] += 1;
        }  
        
        fevskip[i_file]==0;
    }
            
}
