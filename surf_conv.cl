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
    int surf_x=*surf_x_b;
    int surf_y=*surf_y_b;
    int n_pol=*n_pol_b;
    int tau=*tau_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;        
    int image_x;
    int image_y;
    __local int lin_idx;
    __local float ts_value;
    __local int xs_i;
    __local int ys_i;
    __local int ps_i;
    __local int ts_i;  
    __local float tmp_ts_value;  



    ts_value=0;

    lin_idx = idx2d(i_file, (int) get_global_size(0), ev_i, n_events);
    xs_i = xs[lin_idx];
    ys_i = ys[lin_idx];
    ps_i = ps[lin_idx];
    ts_i = ts[lin_idx];   
    if (ts_i!=-1){//Zeropad events here are actually -1 padded
        lin_idx = idx4d(i_file, (int) get_global_size(0), xs_i, res_x, ys_i, res_y,
                        ps_i, n_pol);
        tcontext[lin_idx] = ts_i;
        if (ts_mask[lin_idx]==0){
            ts_mask[lin_idx]=1;}
            
        //Actual relative indices           
        image_x = xs_i+(rel_x-surf_x/2);
        image_y = ys_i+(rel_y-surf_x/2);
        
        if (image_x>=0 && image_y>=0 && image_x<=res_x && image_y<=res_y){   //zeropad would make this faster   
            lin_idx = idx4d(i_file, (int) get_global_size(0), image_x, res_x, 
                            image_y, res_y, ps_i, n_pol);
            //Test, then continue here
            if (ts_mask[lin_idx]==1){
                tmp_ts_value = exp(  ((float)(tcontext[lin_idx]-ts_i)) / (float)tau );
                if (tmp_ts_value>0 && tmp_ts_value<1){//doublecheck overflowing
                    ts_value=tmp_ts_value;
                }
                    
                }
                
            }       
        }
        
    //No polarities for now
    lin_idx = idx4d(i_file, (int) get_global_size(0), ev_i, n_events,
                                            rel_x, surf_x, rel_y, surf_y);

    TS[lin_idx] = ts_value;
    
    if(i_file==0 && rel_x==0 && rel_y==0){
        *ev_i_b=ev_i+1;}
    
}
