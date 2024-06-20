
def build_args():

    args = {}
    args['device'] = -1 
    args['lr'] = 0.01
    args['weight_decay'] = 5e-4
    args['negative_slope'] = 0.2
    args['mask_rate'] = 0.5
    args['alpha_l'] = 3
    args['optimizer'] = 'adam'
    args['loss_fn'] = "sce"
    args['pooling'] = "mean"
    return args