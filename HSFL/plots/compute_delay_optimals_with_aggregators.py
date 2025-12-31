def compute_csfl_delay(h,v,V,layer_mflops_one_sample,layer_mbytes,R,N,samples,p_n,p_k,p_s,mylambda):
    D0 = max(
        sum(layer_mbytes[:h]) / R,  #MBYTES/MBYTES
        sum(layer_mbytes[h:v]) / R  #MBYTES/MBYTES
    )
    term1_1=(samples * sum(layer_mflops_one_sample[:h]))*10**6 / p_n
    term1_2=layer_mbytes[h+1]/R
    aggregators=round(mylambda*N)
    CAR= (N-aggregators)/aggregators # 100-5=95/5
    term1_3= ((samples * sum(layer_mflops_one_sample[h:v])) * CAR)*10**6/p_k
    term1_4=(layer_mbytes[v] )/R
    D1 = term1_1+term1_2+term1_3+term1_4
    term1 = ((2 * N * samples*sum(layer_mflops_one_sample[v:V+1])))*10**6 / p_s
    term2_1=term1_3
    term2_2=term1_4
    term2_3=term1_1
    D2 = max(term1, term2_1+term2_2+term2_3)
    D3 = max(
        sum(layer_mbytes[:h]) / R,
        sum(layer_mbytes[h:v]) / R
    )
    D_round = D0 + (D1 + D2) + D3
    return D_round

def find_optimal_h_v_csfl(V, layer_mflops_one_sample, layer_mbytes, R, N, samples, p_n, p_k, p_s, mylambda):
    best_h = None
    best_v = None
    min_delay = float('inf')

    # Iterate over all possible values of h and v
    for h in range(2, V-2):  # h should be between 1 and 6
        for v in range(h + 1, V-1):  # v should be between h+1 and 7
            delay = compute_csfl_delay(h, v, V, layer_mflops_one_sample, layer_mbytes, R, N, samples, p_n, p_k, p_s, mylambda)

            if delay < min_delay:
                min_delay = delay
                best_h, best_v = h, v
    return best_h, best_v, min_delay

def compute_overhead(N,lamda):
    total_size=131 #mb
    av=2  #cut layer parameters (in MB)
    ah=1 #collaborative layer parameters (in MB)
    B=19 #number of batches
    cm=0.5*total_size #client model's parameters (in MB)
    am=0.3*total_size #aggregators model's parameters (in MB)
    wm=0.2*total_size #weak model's parameters (in MB)

    comm_per_round_sfl=((2*av*B + 2 * cm) * N ) / 1000 #convert MB to GB

    return comm_per_round_sfl #it is in GB

if __name__ == "__main__":

    layer_names=['conv1','conv2','conv3','conv4','conv5','conv6','conv7','conv8','fc1','fc2','fc3']
    #layer_mflops_one_sample=[239,474,944,1900,3800,7600,7600,7600,7600,204,32,1] #it is in FLOPS, for forward and backward propagation
    #old_one
    #layer_mflops_one_sample=[80,150,330,620,1250,2500,2500,2500,2500,80,32,1] #it is in FLOPS, for forward and backward propagation
    #old_one
    layer_mflops_one_sample=[0.1,0.76,1.57,3.14,2.36,4.72,4.72,4.72,4.2,33.6,0.1] #for MNIST dataset
    layer_mflops_one_sample= [i * 2 for i in layer_mflops_one_sample] # multiply by 2 for forward and backward propagation
    #layer_mbytes=[46,23,10,12,6.5,8.3,4.8,4.8,108,16.6,0.2]# It is in Mbytes, for transmission
    layer_mbytes=[10,10,10,10,10,10,10,10,10,10,10]# It is in Mbytes, for transmission
    R=10 # Mbps
    N=100
    samples= 60000/N
    pn=1.5 * 4 *2#Ghz (ghz per core per 2 Flops/cycle)
    pk=2.8 * 8 * 2#Ghz
    ps=100 #Ghz
    mylambda=0.25
    p_n = pn * 10**9  # GHz converted to Hz
    p_k = pk * 10**9  # GHz converted to Hzz
    p_s = ps * 10**9  # GHz converted to Hz
    V=11
    optimal_h, optimal_v, min_delay_csfl = find_optimal_h_v_csfl(V, layer_mflops_one_sample, layer_mbytes, R, N, samples, p_n, p_k, p_s, mylambda)
    comm_overhead_csfl=compute_overhead(N,mylambda)
    print(compute_csfl_delay(optimal_h,optimal_v,V,layer_mflops_one_sample,layer_mbytes,R,N,samples,p_n,p_k,p_s,mylambda))
    print("Optimal layers for C-SFL")
    print(f"Optimal (h, v): ({optimal_h}, {optimal_v}) with minimum CSFL delay: {min_delay_csfl:.6f} seconds + communication overhead {comm_overhead_csfl}")