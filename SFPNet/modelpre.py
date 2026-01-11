
from model_setup import SPFNet_A_inputs,SPFNet_F_inputs

def prepare_model(MODEL, FLAG, data,train_gt, val_gt, test_gt,
                  train_onehot, val_onehot, test_onehot, class_count, device):
    learning_rate = 5e-4
    WEIGHT_DECAY = 0
    max_epoch = 500

    net_input, train_gt_tensor, val_gt_tensor, test_gt_tensor = None, None, None, None
    train_onehot_tensor, val_onehot_tensor, test_onehot_tensor = None, None, None
    net = None



    if MODEL == 'SPFNetA':
        if FLAG in [1, 4, 5]:
            superpixel_scale = 200
        else:
            superpixel_scale = 300
        net_input, \
        train_gt_tensor, val_gt_tensor, test_gt_tensor, \
        train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
        net = SPFNet_A_inputs(
            data,train_gt, val_gt, test_gt,
            train_onehot, val_onehot, test_onehot,
            class_count, superpixel_scale, device,FLAG)

    elif MODEL == 'SPFNetF':
        if FLAG in [1, 4, 5]:
            superpixel_scale = 200
        else:
            superpixel_scale = 300
        net_input, \
        train_gt_tensor, val_gt_tensor, test_gt_tensor, \
        train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
        net = SPFNet_F_inputs(
            data,train_gt, val_gt, test_gt,
            train_onehot, val_onehot, test_onehot,
            class_count, superpixel_scale, device,FLAG)
    else:
        None



    return net_input, \
           train_gt_tensor, val_gt_tensor, test_gt_tensor, \
           train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
           net,\
           learning_rate, WEIGHT_DECAY, max_epoch
