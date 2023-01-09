
__kernel void next_ev(__global int *ev_i_b)
{
    if(get_global_id(0)==0){
        *ev_i_b=*ev_i_b+1;
    }     
}
