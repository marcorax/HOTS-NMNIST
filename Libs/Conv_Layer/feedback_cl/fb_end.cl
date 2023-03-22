//Transform an address in Nd dimensions to linear dimensions.
#define idx5d(a,al,b,bl,c,cl,d,dl,e,el) a*bl*cl*dl*el + b*cl*dl*el + c*dl*el + d*el + e
#define idx4d(a,al,b,bl,c,cl,d,dl) a*bl*cl*dl + b*cl*dl + c*dl + d
#define idx3d(a,al,b,bl,c,cl) a*bl*cl + b*cl + c 
#define idx2d(a,al,b,bl) a*bl + b

__kernel void fb_end(__global int *ts, __global int *ev_i_b,
                     __global int *n_events_b, __global int *closest, 
                     __global int *batch_labels,__global double *partial_sum,
                     __global float *S, __global float *dS, 
                     __global int *bevskip)
{                          
    int i_file = (int) get_global_id(0);
    int nfiles = (int) get_global_size(0);
    int lid = (int) get_local_id(1); 
    int lsize = (int) get_local_size(1);
    
    int ev_i=*ev_i_b;
    int n_events=*n_events_b;    

    int lin_idx;
    int ts_i;  
    int loc_idx;
    
    float tmp_S;
    
    

    lin_idx = idx2d(i_file, nfiles, ev_i, n_events);

    ts_i = ts[lin_idx];   
        
    if (ts_i!=-1 && bevskip[i_file]==0){//Zeropad events here are actually -1 padded

        loc_idx = idx2d(i_file, nfiles, lid, lsize);
                        
        tmp_S = (float) work_group_reduce_add(partial_sum[loc_idx]);
        partial_sum[loc_idx]=0; //reset fpartial sum for next event
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (get_local_id(1)==0){
        
            if (closest[i_file]!=batch_labels[i_file]){
            tmp_S = -tmp_S;}
        
            if(ev_i==0){
                dS[i_file] = tmp_S;
            }
            else{
                dS[i_file] = tmp_S-S[i_file];
            }
            
            S[i_file] = tmp_S;
        
        }
            
    }
            
}
