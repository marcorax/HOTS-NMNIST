//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void fb_context_update(__global int *ts, __global int *n_clusters_b,
                                __global int *ev_i_b, __global int *n_events_b,
                                __global int *tcontext, __global int *ts_mask,
                                __global int *closest, __global int *bevskip)
{                          
    int i_file = (int) get_global_id(0);
    int nfiles = (int) get_global_size(0);
    
    int n_clusters=*n_clusters_b;   
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    

    int lin_idx;
    int ts_i;  
    
        
    lin_idx = idx2d(i_file, nfiles, ev_i, n_events);

    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && bevskip[i_file]==0){//Zeropad events here are actually -1 padded

        lin_idx = idx2d(i_file, nfiles, closest[i_file], n_clusters);
                        
        tcontext[lin_idx] = ts_i;
        
        if (ts_mask[lin_idx]==0){
            ts_mask[lin_idx]=1;}
            
 
    }
            
}
