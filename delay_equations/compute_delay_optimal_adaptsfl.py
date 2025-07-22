



if __name__ == "__main__":
    
    #Model parameters
    layer_names=['conv1','conv2','conv3','conv4','conv5','conv6','conv7','conv8','fc1','fc2','fc3']#for CIFAR-10 dataset
    #layer_names=['conv1','conv2','conv3','conv4','conv5','fc1','fc2','fc3']#for MNIST dataset

    #We work with MFLOPS
    layer_mflops_one_sample=[3.54,37,37,75,37,75,19,19,205,33,1] #for CIFAR-10 dataset
    #layer_mflops_one_sample=[1,7.25,7,5,2,1,1,1] #for MNIST dataset

    #We work with Mbytes
    layer_mbytes=[1,10,10,10,10,10,10,10,400,67,10]#for CIFAR-10 dataset
    #layer_mbytes=[1,7,8,7,10,80,30,10]#for MNIST dataset
    
    #Topology parameters
    R=20 # Mbps
    N=100
    samples= 50000/N
    
    #Server and device parameters
    pn=1.5 * 4 *2 #Ghz (ghz per core per 2 Flops/cycle)
    pk=2.8 * 8 * 2 #Ghz
    ps=100 #Ghz
    p_n = pn * 10**9  # GHz converted to Hz
    p_k = pk * 10**9  # GHz converted to Hz
    p_s = ps * 10**9  # GHz converted to Hz

    #Number of layers
    V=11
   
    print("Optimal layers for SFL")
    #optimal_v,min_delay_sfl=find_optimal_h_v_sfl(V, layer_mflops_one_sample, layer_mbytes, R, N, samples, p_n, p_k, p_s)
    print(f"ADAPTSFL optimal layer:({optimal_v}),delay: {min_delay_sfl:.6f} seconds")