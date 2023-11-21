import torch
from gem_attention import GeMAttention, geometric_mean, harmonic_mean, generalized_mean



def attention_min():

    ## INPUT DATA (FAILS WITH NEGATIVE VALUES)
    query = torch.zeros((1, 1, dim))
    context = torch.Tensor([[
        [1, 0.1, 5,  -1],
        [2, 0.2, 6,  -2],
        [3, 0.3, 19, -3],
        [4, 0.4, 34, -4],
    ]])
    min_ = context.min(dim=1)[0]

    ## SET WEIGHTS AND PARAMETER P
    dim=4
    attn_min = GeMAttention(query_dim=dim, context_dim=dim, hidden_dim=dim, num_heads=1)
    ## AGGREGATION PARAMETER
    with torch.no_grad(): attn_min.p[:] = -1e+6
    ## SHIFT
    with torch.no_grad(): attn_min.V.bias[:] = 5
    with torch.no_grad(): attn_min.out.bias[:] = -5

    ## COMPUTE MIN
    a_min = attn_min(query=query, context=context)
    assert torch.all(a_min - min_ < 1e-4) 
    return 



def attention_max():

    ## INPUT DATA (FAILS WITH NEGATIVE VALUES)
    query = torch.zeros((1, 1, dim))
    context = torch.Tensor([[
        [1, 0.1, 5,  -1],
        [2, 0.2, 6,  -2],
        [3, 0.3, 19, -3],
        [4, 0.4, 34, -4],
    ]])
    max_ = context.max(dim=1)[0]

    ## SET WEIGHTS AND PARAMETER P
    dim=4
    attn_max = GeMAttention(query_dim=dim, context_dim=dim, hidden_dim=dim, num_heads=1)
    ## AGGREGATION PARAMETER
    with torch.no_grad(): attn_max.p[:] = 1e+6
    ## SHIFT
    with torch.no_grad(): attn_max.V.bias[:] = 5
    with torch.no_grad(): attn_max.out.bias[:] = -5

    ## COMPUTE MIN
    a_max = attn_max(query=query, context=context)
    assert torch.all(a_max - max_ < 1e-4) 
    return 



def attention_mean():

    ## INPUT DATA (FAILS WITH NEGATIVE VALUES)
    query = torch.zeros((1, 1, dim))
    context = torch.Tensor([[
        [1, 0.1, 5,  -1],
        [2, 0.2, 6,  -2],
        [3, 0.3, 19, -3],
        [4, 0.4, 34, -4],
    ]])
    mean = context.mean(dim=1)

    ## SET WEIGHTS AND PARAMETER P
    dim=4
    attn_mean = GeMAttention(query_dim=dim, context_dim=dim, hidden_dim=dim, num_heads=1)
    ## AGGREGATION PARAMETER
    with torch.no_grad(): attn_mean.p[:] = 1
    ## SHIFT
    with torch.no_grad(): attn_mean.V.bias[:] = 5
    with torch.no_grad(): attn_mean.out.bias[:] = -5

    ## SHIFT AND COMPUTE MIN
    a_mean = attn_mean(query=query, context=context)
    assert torch.all(a_mean - mean < 1e-5) 
    return



def attention_std():

    ## TWO ATTENTION LAYERS
    ## FIRST COMPUTES MEAN, SECOND USES MEAN TO COMPUTE STD
    dim = 4

    ## SET WEIGHTS AND PARAMETER P
    attn_mean = GeMAttention(query_dim=dim, context_dim=dim, hidden_dim=dim, num_heads=1)
    with torch.no_grad(): attn_mean.p[:] = 1
    with torch.no_grad(): attn_mean.out.weight[:] *= -1
    with torch.no_grad(): attn_mean.V.bias[:] = 5

    ## SET WEIGHTS AND PARAMETER P
    attn_std = GeMAttention(query_dim=4, context_dim=4, hidden_dim=4, num_heads=1)
    ## AGGREGATION PARAMETER
    with torch.no_grad(): attn_std.p[:] = 2
    ## SHIFT
    with torch.no_grad(): attn_std.V.bias[:] = 5

    ## INPUT DATA (FAILS WITH NEGATIVE VALUES)
    query = torch.zeros((1, 1, dim))
    context = torch.Tensor([[
        [1, 0.1, 5,  -1],
        [2, 0.2, 6,  -2],
        [3, 0.3, 19, -3],
        [4, 0.4, 34, -4],
    ]])
    mean = context.mean(dim=1)
    population_std = context.std(dim=1, correction=0)

    # ## COMPUTE MEAN with GeM 
    # a_mean = attn_mean(query=query, context=context)
    # assert torch.all(-a_mean[..., :-1] == mean[..., :-1])
    
    # ## COMPUTE POPULATION STD with GeM 
    # a_std = attn_std(query=query, context=context + a_mean)
    # population_std = context.std(dim=1, correction=0)
    # assert torch.all(a_std[..., :-1] - population_std[..., :-1] < 1e-5)

    # # ## COMPUTE SAMPLE STD GeM -> not possible? weights would have to add to greater than 1
    # # a_std = attn_std(query=query, context=context + a_mean)
    # # sample_std = context.std(dim=1)
    # # assert torch.all(a_std == sample_std)

    ## TODO: COMPUTE WITH NEGATIVE NUMBERS -> Just shift the context by some upper bound constant
    ## Only works if we take abs(v) not clamp(v)
    ## COMPUTE MIN and SUBTRACT FROM CONTEXT -> min to zero
    a_mean = attn_mean(query=query, context=context)
    a_std = attn_std(query=query, context=context + a_mean)
    assert torch.all(a_std - population_std < 1e-5) 
    ## ADD MIN BACK TO RESULT


    ## CHECK NONLINEAR MEANS UNDER SHIFT
    (context.pow(2).sum(dim=1)/4).pow(0.5)
    x = context[..., -1]
    
    geometric_mean(x)
    geometric_mean(x-x.min()) + x.min()
    geometric_mean(x.abs()) * x.prod().sign()
    geometric_mean(x[...,:-1].abs()) * x[...,:-1].prod().sign()

    harmonic_mean(x)
    harmonic_mean(x-x.min())

    ## SHOULD I MIN SHIFT TO MAKE EVERYTHING POSITIVE? 
    ## NO, p ≤ 0 always maps to zero

    ## SHOULD I COMPUTE THE MEAN ON abs(x)? 
    
    
    ## SHOULD I IGNORE NON-POSITIVE VALUES clamp(x, eps)? 
    ## Lose ability to compute true std and stuff without shifting

    ## SHOULD I SHIFT BY THE MINIMUM + eps?

    ## TODO: failure cases
    ##  • x = negative -> nans when p = 1/even
    ##  • p = zero

    ## IF p is a rational number with an even denominator, add a small

    ## NOTE: Okay cases
    ##  • x = zero, p = negative -> torch sends exp(0, -p) -> inf and resolves nicely to zero
    generalized_mean(torch.Tensor([0,1,2]), p=-3)
    return



def attention_amplitude():

    attn = GeMAttention

    ## SET QKV WEIGHTS AND PARAMETER P

    return



def attention_dist():

    attn = GeMAttention

    ## SET QKV WEIGHTS AND PARAMETER P

    return



def attention_amplitude():

    attn = GeMAttention

    ## SET QKV WEIGHTS AND PARAMETER P

    return



def attention_loss():

    ## INPUT DATA (FAILS WITH NEGATIVE VALUES)
    context = torch.Tensor([[
        [1.0, 2.0, 3.0, 4.0],  ## x 
        [4.7, 0.0, 0.0, 0.0],  ## f(x)
        # [0.3, 1.0, 0.6, 2.8],   ## f(x)
        [3.0, 0.3, 19., -3.],  ## x 
        [2.75, 0.0, 0.0, 0.],  ## f(x)
        # [0.9, .15, 3.8, -2.1],  ## f(x)
        [4.0, 0.4, 34., -4.],  ## x
        [5.4, 0.0, 0.0, 0.0],  ## f(x)
        # [1.2, 0.2, 6.8, -2.8],  ## f(x)
    ]])
    dim = context.shape[-1]
    query = torch.zeros((1, 1, dim))
    ## WANT SOMETHING IN THE TOKENS TO INDICATE TYPE AND PAIRING (zeros indicate type)

    ## SET WEIGHTS AND PARAMETER P
    attn_loss = GeMAttention(query_dim=dim, context_dim=dim, hidden_dim=dim, num_heads=1)
    ## AGGREGATION PARAMETER
    with torch.no_grad(): attn_loss.p[:] = 2
    ## SHIFT
    with torch.no_grad(): attn_loss.V.weight[0] = torch.Tensor([0.3, 0.5, 0.2, 0.7]) # f(x)
    with torch.no_grad(): attn_loss.V.weight[1:, :] = 0
    with torch.no_grad(): attn_loss.V.bias[:] = 50
    # with torch.no_grad(): attn_loss.Q.weight[:, 0] = torch.Tensor([1, 1, 1, 1]) # f(x)
    # with torch.no_grad(): attn_loss.Q.weight[1:, :] = 0.0
    with torch.no_grad(): attn_loss.K.weight[0] = torch.Tensor([0.3, 0.5, 0.2, 0.7]) # f(x)
    with torch.no_grad(): attn_loss.K.weight[1:, :] = 0
    with torch.no_grad(): attn_loss.out.bias[:] = -50
    with torch.no_grad(): attn_loss.out.weight[:] *= -1


    ## ALGORITHM 
    ## Have each f(x) attend to its V(x)
    ## Add the negative of the f(x) to the token
    ## Compute the RMS of the difference = loss

    ## COMPUTE MIN
    a_loss = attn_loss(query=context, context=context)
    # assert torch.all(a_min - min_ < 1e-4) 

    return





if __name__ == "__main__":

    attention_loss()
    attention_std()







