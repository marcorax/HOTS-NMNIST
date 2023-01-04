#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void class_infer(__global int *xs,__global int *ys, __global int *ts,
                         __global int *res_x_b, __global int *res_y_b, 
                         __global int *tau_b, __global int *n_pol_b, 
                         __global int *n_clusters_b, __global int *ev_i_b,
                         __global int *n_events_b, __global int *tcontext,
                         __global int *ts_mask, __global float *weights,
                         __global float *distances, __global int *closest_prev,
                         __global int *closest)
{
    unsigned int i_file = get_global_id(0); 
    int n_clusters=*n_clusters_b;   
    int res_x=*res_x_b;
    int res_y=*res_y_b;
    int n_pol=*n_pol_b;
    int tau=*tau_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;  
    unsigned int image_x = get_local_id(1);
    unsigned int image_y = get_local_id(2)%res_y; 
    unsigned int image_p = get_local_id(2)/res_y; 

          
    float ts_value; // default allocation is private, faster than local
    float tmp_ts_value;  
    __local int lin_idx;
    __local int xs_i;
    __local int ys_i;
    __local int ps_i;
    __local int ts_i;  
    float elem_distance; 
    float min_distance;

    //continue here
    ts_value=0;

    lin_idx = idx2d(i_file, (int) get_global_size(0), ev_i, n_events);
    xs_i = xs[lin_idx];
    ys_i = ys[lin_idx];
    ps_i = *closest_prev;
    ts_i = ts[lin_idx];   
    if (ts_i!=-1){//Zeropad events here are actually -1 padded
        lin_idx = idx4d(i_file, (int) get_global_size(0), xs_i, res_x, ys_i, res_y,
                        ps_i, n_pol);
        tcontext[lin_idx] = ts_i;
        if (ts_mask[lin_idx]==0){
            ts_mask[lin_idx]=1;}
                    
        lin_idx = idx4d(i_file, (int) get_global_size(0), image_x, res_x, 
                        image_y, res_y, image_p, n_pol);
                        
        //Test, then continue here
        if (ts_mask[lin_idx]==1){
            tmp_ts_value = exp(  ((float)(tcontext[lin_idx]-ts_i)) / (float)tau );
            if (tmp_ts_value>0 && tmp_ts_value<1){//doublecheck for overflowing
                ts_value=tmp_ts_value;
                
                    
            }
                
        }       
    }  

    
    for (int cl=0; cl<n_clusters; cl++){
    
        lin_idx = idx4d(i_file, (int) get_global_size(0), image_x, res_x,
                         get_local_id(2), res_y*n_pol, cl, n_clusters);
    
        //Euclidean is causing a good chunk of errors, moving to L1 
        elem_distance = fabs(weights[lin_idx]-ts_value);

            
        lin_idx = idx3d(i_file, (int) get_global_size(0), ev_i, n_events, cl,
                         n_clusters);                                            
                         
        distances[lin_idx] = work_group_reduce_add(elem_distance);
        
        if (get_local_id(1)==0 && get_local_id(2)==0){
        
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
    
    
    if(i_file==0 && get_local_id(1)==0 && get_local_id(2)==0){
        *ev_i_b=ev_i+1;
    }
    
}
