import itertools


def compute_latency_adaptsfl(v,V,layer_mflops_one_sample,layer_mbytes,R,N,samples,p_n,p_k,p_s):
    I=3 #epochs per aggregation
    print("cut layer:"+str(v))
    
    #Clients download model from the server
    #D0 = max( sum(layer_mbytes[:v]) / R,  0 )
    #print("client model tx delay:"+str(D0))

    #Clients perform FP
    term1_1=(samples*( sum(layer_mflops_one_sample[:v]))*10**6 / p_n)
    print("fp delay:"+str(term1_1))

    #Clients transmit cut layer activations
    term1_2=layer_mbytes[v]/R
    print("cut layer tx delay:"+str(term1_2))

    #Clients perform BP (double the FP)
    term1_3=2*term1_1
    print("bp delay:"+str(term1_3))

    #Server performs FP and BP from layer v to layer V
    term1 = ((2 * N * samples*sum(layer_mflops_one_sample[v:V+1])))*10**6 / p_s
    print("server delay:"+str(term1))

    #Server transmits to clients the cut layer activations
    term1_2=layer_mbytes[v]/R
    print("server cut layer tx delay:"+str(term1_2))
    D2=term1_1+term1_2+term1+term1_2+term1_3+term1_2

    #Clients upload model to the server
    #D3 = D0
    #print("client model tx delay:"+str(D3))
    D_round = D2
    print("total delay:"+str(D_round))
    return D_round

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