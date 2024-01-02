import torch


def generate_batch(num_items=10, num_features=3, batch_size=32, device="cpu"):

    # num_functions = 4
    num_functions = 3
    batch_size = batch_size // num_functions

    # vector x
    # [0.9724, 0.8452, 0.8948]
    # sample, func
    # [x, y, z], max
    # target
    # [max(x,y,z)]

    features = torch.rand((batch_size, num_items, num_features))

    max_pool = features.max(dim=1)[0]
    max_pool_func = torch.ones((len(max_pool))) * 0

    min_pool = features.min(dim=1)[0]
    min_pool_func = torch.ones((len(min_pool))) * 1

    avg_pool = features.mean(dim=1)
    avg_pool_func = torch.ones((len(avg_pool))) * 2

    # sum_pool = features.sum(dim=1)
    # sum_pool_func = torch.ones((len(sum_pool))) * 3

    # ## SELECT THE VECTOR WITH THE MAX MAGNITUDE
    # argmax = features.norm(dim=2).argmax(dim=1)
    # max_magn = features[:, argmax].diagonal().T
    # ## SELECT THE VECTOR WITH THE MIN MAGNITUDE
    # min_magn
    # ## SELECT THE VECTOR WITH THE MEDIAN MAGNITUDE
    # med_magn


    # min_dist = features.dist(features[:, 0:1], dim=0).min
    # max_dist
    # avg_dist
    # sum_dist

    features = torch.tile(features, (num_functions,1,1)).to(device)

    functions = torch.cat((
        max_pool_func, 
        min_pool_func, 
        avg_pool_func,
        # sum_pool_func,
        # max_magn_func,
        # min_magn_func,
        # avg_magn_func,
        # sum_magn_func,
        # min_dist_func,
        # max_dist_func,
        # avg_dist_func,
        # sum_dist_func,
    )).unsqueeze(-1).to(device)

    targets = torch.cat((
        max_pool, 
        min_pool, 
        avg_pool,
        # sum_pool,
        # max_magn,
        # min_magn,
        # avg_magn,
        # sum_magn,
        # min_dist,
        # max_dist,
        # avg_dist,
        # sum_dist,
    ), dim=0).to(device)

    return features, functions, targets


def generate_test(device):


    features = torch.Tensor([
        [[0.0, 0.5, 1.0],
         [1.0, 1.0, 1.0]],

        [[0.0, 0.25, 0.0],
         [0.5, 0.75, 0.0]],
    ])
    max_pool = features.max(dim=1)[0]
    min_pool = features.min(dim=1)[0]
    avg_pool = features.mean(dim=1)
    # sum_pool = features.sum(dim=1)


    features = torch.tile(features, (3,1,1)).to(device)

    functions = torch.Tensor([
        [0,0,1,1,2,2]
        # [0,0,1,1,2,2,3,3]
    ]).T.to(device)

    targets = torch.cat((
        max_pool, 
        min_pool, 
        avg_pool,
        # sum_pool,
    ), dim=0).to(device)

    return features, functions, targets


def generate_means(device):

    features = torch.Tensor([
        [[0, 0, 0.0, 0, 0],
         [1, 1, 1.0, 1, 1],
         [2, 2, 2.0, 2, 2],
         [3, 3, 3.0, 3, 3],
         [4, 4, 4.0, 4, 4]],
    ])
    # max_pool = features.max(dim=1)[0]
    # min_pool = features.min(dim=1)[0]
    avg_pool = features.mean(dim=1)

    features = torch.tile(features, (1,1,1)).to(device)

    functions = torch.Tensor([
        [2]
        # [2,2,2]
    ]).T.to(device)

    targets = torch.cat((
        # max_pool, 
        # min_pool, 
        avg_pool,
    ), dim=0).to(device)

    return features, functions, targets
