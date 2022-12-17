#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx2d(a,al,b,bl) a*bl + b

__kernel void all_surfaces(__global int *xs,__global int *ys,__global int *ps,
                      __global int *ts, __global int *res_x_b,
                      __global int *res_y_b, __global int *tau_b,
                       __global int *n_pol_b, __global float *TS, 
                       __global int *n_events_b)
{
    unsigned int i_file = get_global_id(0);
    unsigned int x = get_local_id(1);
    unsigned int y = get_local_id(2);        
    int n_events = *n_events_b;
    int res_x=*res_x_b;
    int res_y=*res_y_b;
    

    int lin_idx;
    for(int i_event = 0; i_event < n_events; i_event = i_event + 1){

        lin_idx = idx4d(i_file, (int) get_global_size(0), i_event, n_events,
                        x, res_x, y, res_y);
        TS[lin_idx] = i_file;
    
    }
}
