import argparse
import torch
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

def histcounts(data, epsilon, min_=-1, max_=1):
    """
    example: data = [0.6054744899487247, 0.6986512231376916, 0.9243823257809534, 0.9308167830778726], epsilon = 10, range(-1, 1)
             [0.6054744899487247, 0.6986512231376916, 0.9243823257809534, 0.9308167830778726] ===> [0. 0. 0. 0. 0. 0. 0. 0. 2. 2.] ===> [0. 0. 0. 0. 0. 0. 0. 0. 0.5. 0.5.]
    params:
        range(min,max)   ===> state_interval
        epsilon ===> epsilon-level split
        data    ===> orbits_cosine_similarity_sequence
    return:
        hist : a list about each interval frequence
        statistical_probabilities_sequence: statistical probabilities of epsilon intervals
    """
    data = data.float()
    hist = torch.histc(data, bins=epsilon, min=min_, max=max_)
    statistical_probabilities_sequence = hist / torch.sum(hist)
    return hist, statistical_probabilities_sequence
    
def DE(statistical_probabilities_sequence, epsilon):
    """
    example: statistical_probabilities_sequence = [0. 0. 0. 0. 0. 0. 0. 0. 2. 2.], epsilon = 10
             [0. 0. 0. 0. 0. 0. 0. 0. 2. 2.] ===> 0.301
    params:
        statistical_probabilities_sequence ===> statistical probabilities of epsilon intervals
        epsilon                            ===> epsilon-level split
    return: DE_value
    """
    # caculate de value
    DE_value = -1 / torch.log(torch.tensor(epsilon)) * torch.nansum(statistical_probabilities_sequence * torch.log(statistical_probabilities_sequence), dim=0)
    
    return DE_value

def calculate_DE(ori_data, embed_size, epsilon):
    """
    example：ori_data = [1, 2, 13, 7, 9, 5, 4], embed_size = 3, epsilon = 10
             [1, 2, 13, 7, 9, 5, 4] ===> 0.9896002614175352
    params：
        ori_data       ===> sequence data
        embedding_size ===> dimension of new space
        epsilon        ===> Divide the interval [-1, 1] into epsilon equal segments.
    return： DE_value
    """
    # build orbits along second dimension, operate token_length-dimension(1,embed_*,*)
    orbits = ori_data.unfold(1, embed_size, 1)  # [1, token_length, samples_size]---> [1, token_length-embed_size+1, samples_size, embed_size]
    # calculate cosine similarity of orbits
    orbits_cosine_similarity_sequence = torch.nn.functional.cosine_similarity(orbits[:, :-1], orbits[:, 1:], dim=-1) # [1, token_length-embed_size+1, samples_size, embed_size]---> [1, token_length-embed_size, samples_size]
    # Placing the cosine similarity into intervals, operate sample_size-dimension(in_dims=-1)
    batched_1 = torch.vmap(histcounts, in_dims=-1, out_dims=1) 
    hist, statistical_probabilities_sequence = batched_1(orbits_cosine_similarity_sequence, epsilon=epsilon)  
    # calculate de
    DE_value = DE(statistical_probabilities_sequence, epsilon)
    # print(DE_value)
    return DE_value

def get_tau_scale_DE(ori_data, embed_size, epsilon, tau):
    """
    example: ori_data = [1, 2, 13, 7, 9, 5, 4], embed_size=3, epsilon = 10,  tau = 2
             [1, 2, 13, 7, 9, 5, 4] ===> [1.5, 7.5, 10.0, 8.0, 7.0, 4.5] ==> de_value([1.5, 7.5, 10.0, 8.0, 7.0, 4.5])
    params:
        ori_data ===> sequence data
        embedding_size ===> dimension of new space
        epsilon        ===> Divide the interval [-1, 1] into epsilon equal segments.
        tau      ===> tau-level sequence of ori_data
    return: tau_scale_de
    """
    # get sub-series
    windows = ori_data.unfold(1, tau, 1) 
    tau_scale_sequence = torch.mean(windows, dim=3) # Pay attention, in this case dim=3
    # caculate tau_scale de value
    de = calculate_DE(tau_scale_sequence, embed_size, epsilon)
    # return de.unsqueeze(0)
    return de

def get_tau_multiscale_DE(ori_data, embed_size, epsilon,  tau_prime):
    """
    example: ori_data = [1, 2, 13, 7, 9, 5, 4], embed_size=3, epsilon = 10,  tau = 3
             [1, 2, 13, 7, 9, 5, 4] ===> {'tau = 1': [1.0, 2.0, 13.0, 7.0, 9.0, 5.0, 4.0], 'tau = 2': [1.5, 7.5, 10.0, 8.0, 7.0, 4.5], 'tau = 3': [5.333333333333333, 7.333333333333333, 9.666666666666666, 7.0, 6.0]}
                                    ===> {'tau = 1': de([1.0, 2.0, 13.0, 7.0, 9.0, 5.0, 4.0]), 'tau = 2': de([1.5, 7.5, 10.0, 8.0, 7.0, 4.5]), 'tau = 3': de([5.333333333333333, 7.333333333333333, 9.666666666666666, 7.0, 6.0])}
                                    ===> [0.30102999566398114, -0.0, -0.0]
                                    ===> max([0.30102999566398114, -0.0, -0.0]) - min([0.30102999566398114, -0.0, -0.0])
    params:
        ori_data ===> sequence data
        embedding_size ===> dimension of new space
        epsilon        ===> Divide the interval [-1, 1] into epsilon equal segments.
        tau_prime ===> multiscale_sequence from tau_1 to tau_prime
    return: mde
    """
    # mde = torch.zeros(tau_prime)
    mde = []
    for temp_tau in range(1,tau_prime + 1):
        value = get_tau_scale_DE(ori_data, embed_size, epsilon, temp_tau)
        mde.append(value) 
    mde = torch.stack(mde, dim=0)
    std_mde = torch.std(mde, dim=0)
    return std_mde

def experiment(ori_data, embed_size, epsilon, tau_prime):
    DE = get_tau_scale_DE(ori_data, embed_size, epsilon, 1)
    MDE = get_tau_multiscale_DE(ori_data, embed_size, epsilon, tau_prime)
    print("DE:",DE)
    print("MDE:",MDE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_data', type=list, default=torch.randn(1,175,1000))
    parser.add_argument('--embed_size', type=int, default=3)
    parser.add_argument('--epsilon', type=int, default=5 * torch.randn(1,175,1000).shape[1])
    parser.add_argument('--tau_prime', type=int, default=5)
    args = parser.parse_args()

    experiment(args.ori_data, args.embed_size, args.epsilon, args.tau_prime)
