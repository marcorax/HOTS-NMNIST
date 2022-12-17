#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void surf_conv(__global int *xs,__global int *ys,__global int *ps,
                          __global int *ts, __global int *res_x_b,
                          __global int *res_y_b, __global int *surf_x_b,
                          __global int *surf_y_b, __global int *tau_b,
                           __global int *n_pol_b, __global float *TS, 
                           __global int *ev_i_b, __global int *n_events_b,
                            __global int *tcontext, __global int *ts_mask)
{
    unsigned int i_file = get_global_id(0);
    unsigned int rel_x = get_local_id(1);
    unsigned int rel_y = get_local_id(2);        
    int res_x=*res_x_b;
    int res_y=*res_y_b;
    int surf_x=*res_x_b;
    int surf_y=*res_y_b;
    int n_pol=*n_pol_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;
    __local int lin_idx;
    __local float ts_value;
    __local int xs_i;
    __local int ys_i;
    __local int ps_i;
    __local int ts_i;  
    
    int image_x;
    int image_y;

    ts_value=0;

    lin_idx = idx2d(i_file, (int) get_global_size(0), ev_i, n_events);
    xs_i = xs[lin_idx];
    ys_i = ys[lin_idx];
    ps_i = ps[lin_idx];
    ts_i = ts[lin_idx];   
    lin_idx = idx4d(i_file, (int) get_global_size(0), xs_i, res_x, ys_i, res_y,
                    ps_i, n_pol);
    tcontext[lin_idx] = ts_i;
    if (ts_mask[lin_idx]==0){
        ts_mask[lin_idx]=1;
        
        //Actual relative indices
        rel_x = rel_x-surf_x/2;
        rel_y = rel_y-surf_y/2;
        
        image_x = xs_i-rel_x;
        image_y = ys_i-rel_y;
        
        if (((image_x)>=0 && (image_y)>=0)){      
            lin_idx = idx4d(i_file, (int) get_global_size(0), image_x, res_x, 
                            image_y, res_y, ps_i, n_pol);
            //Test, then continue here
            ts_value = tcontext[lin_idx]*ts_mask[lin_idx];
            
        }
    
    
    }
    
    //No polarities for now
    lin_idx = idx4d(i_file, (int) get_global_size(0), ev_i, n_events,
                                             xs_i, res_x, ys_i, res_y);
    TS[lin_idx] = ts_value;
    
    
}
