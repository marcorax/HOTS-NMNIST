//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void context_update(__global int *xs,__global int *ys,
                             __global int *ps, __global int *ts,
                             __global int *res_x_b, __global int *res_y_b,
                             __global int *win_l_b,
                             __global int *n_pol_b, __global int *ev_i_b,
                             __global int *n_events_b, __global int *tcontext,
                             __global int *ts_mask, __global int *fevskip)
{
    int i_file = (int) get_global_id(0);
    int nfiles = (int) get_global_size(0);
    
    
    int res_x=*res_x_b;
    int res_y=*res_y_b;
    int win_l=*win_l_b;
    int n_pol=*n_pol_b;
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    
   
    int lin_idx;
    int xs_i;
    int ys_i;
    int ps_i;
    int ts_i;  

    lin_idx = idx2d(i_file, nfiles, ev_i, n_events);
    
    //Zeropad indices and res variables
    int pad_x = win_l/2;
    int pad_y = win_l/2;
    res_x = res_x+win_l-1;
    res_y = res_y+win_l-1;

    
    xs_i = xs[lin_idx]+pad_x;//zeropad index
    ys_i = ys[lin_idx]+pad_y;//zeropad index
    ps_i = ps[lin_idx];
    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && fevskip[i_file]==0){//Zeropad events here are actually -1 padded
                     
        lin_idx = idx4d(i_file, nfiles, xs_i, res_x, ys_i, res_y, ps_i, n_pol);
                        
        tcontext[lin_idx] = ts_i;
    
        if (ts_mask[lin_idx]==0){
            ts_mask[lin_idx]=1;}
        
  
    }
}
