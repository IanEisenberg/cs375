from dldata.metrics import utils
import numpy as np

def post_process_neural_regression_msplit_preprocessed(result, ne_path):
    """ 
    Loads the precomputed noise estimates and normalizes the results
    """
    ne = np.load(ne_path)
    sarrays = []
    for s_ind, s in enumerate(result['split_results']):
        farray = np.asarray(
                result['split_results'][s_ind]['test_multi_rsquared'])
        sarray = farray / ne[0]**2
        sarrays.append(sarray)
    sarrays = np.asarray(sarrays)
    result['noise_corrected_multi_rsquared_array_loss'] = \
            (1 - sarrays).mean(0)
    result['noise_corrected_multi_rsquared_array_loss_stderror'] = \
            (1 - sarrays).std(0)
    result['noise_corrected_multi_rsquared_loss'] = \
            np.median(1 - sarrays, 1).mean()
    result['noise_corrected_multi_rsquared_loss_stderror'] = \
            np.median(1 - sarrays, 1).std()


def post_process_neural_regression_msplit(dataset, 
                                          result, 
                                          spec, 
                                          n_jobs=1, 
                                          splits=None, 
                                          nan=False):
    """
    Computes noise estimates and normalizes the results
    """
    name = spec[0]
    specval = spec[1]
    assert name[2] in ['IT_regression', 
                       'V4_regression', 
                       'ITc_regression', 
                       'ITt_regression'], name
    if name[2] == 'IT_regression':
        units = dataset.IT_NEURONS
    elif name[2] == 'ITc_regression':
        units = hvm.mappings.LST_IT_Chabo
    elif name[2] == 'ITt_regression':
        units = hvm.mappings.LST_IT_Tito
    else:
        units = dataset.V4_NEURONS

    units = np.array(units)
    if not splits:
        splits, validations = utils.get_splits_from_eval_config(specval, dataset)

    sarrays = []
    for s_ind, s in enumerate(splits):
        ne = dataset.noise_estimate(
                s['test'], units=units, n_jobs=n_jobs, cache=True, nan=nan)
        farray = np.asarray(
                result['split_results'][s_ind]['test_multi_rsquared'])
        sarray = farray / ne[0]**2
        sarrays.append(sarray)
    sarrays = np.asarray(sarrays)
    result['noise_corrected_multi_rsquared_array_loss'] = \
            (1 - sarrays).mean(0)
    result['noise_corrected_multi_rsquared_array_loss_stderror'] = \
            (1 - sarrays).std(0)
    result['noise_corrected_multi_rsquared_loss'] = \
            np.median(1 - sarrays, 1).mean()
    result['noise_corrected_multi_rsquared_loss_stderror'] = \
            np.median(1 - sarrays, 1).std()


def loss_withcfg(output, *args, **kwargs):
    cfg_dataset = kwargs.get('cfg_dataset', {})
    depth_norm = kwargs.get('depth_norm', 8000)
    label_norm = kwargs.get('label_norm', 20)
    depthloss = kwargs.get('depthloss', 0)
    normalloss = kwargs.get('normalloss', 0)
    ret_dict = kwargs.get('ret_dict', 0)
    nfromd = kwargs.get('nfromd', 0)
    trainable = kwargs.get('trainable', 0)
    multtime = kwargs.get('multtime', 1)
    combine_dict = kwargs.get('combine_dict', 0)
    print_loss = kwargs.get('print_loss', 0)
    extra_feat = kwargs.get('extra_feat', 0)
    
    now_indx = 0
    loss_list = []
    loss_keys = []
    arg_offset = 0
    if cfg_dataset.get('scenenet', 0)==1:
        if ret_dict==1:
            tmp_loss_list = []

        if cfg_dataset.get('scene_normal', 1)==1:
            if nfromd==0:
                curr_loss = normal_loss(output[now_indx], args[now_indx + arg_offset], normalloss = normalloss)
            else:
                curr_loss = normal_loss(output[now_indx], get_n_from_d(args[now_indx + arg_offset]), normalloss = normalloss)
                if cfg_dataset.get('scene_depth', 1)==1:
                    arg_offset = arg_offset - 1

            if trainable==1:
                if normalloss>=1:
                    curr_loss = curr_loss + 1.2
                curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_scene_normal')
            if ret_dict==1:
                tmp_loss_list.append(curr_loss)
            else:
                loss_list.append(curr_loss)
            #loss_keys.append('scene_normal')
            now_indx = now_indx + 1

        if cfg_dataset.get('scene_depth', 1)==1:
            curr_loss = depth_loss(output[now_indx], args[now_indx + arg_offset], depthloss = depthloss, depth_norm = depth_norm)

            if trainable==1:
                curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_scene_depth')

            if ret_dict==1:
                tmp_loss_list.append(curr_loss)
            else:
                loss_list.append(curr_loss)
            #loss_keys.append('scene_depth')
            now_indx = now_indx + 1

        if cfg_dataset.get('scene_instance', 0)==1:
            curr_loss = get_semantic_loss(curr_predict = output[now_indx], curr_truth = args[now_indx + arg_offset], need_mask = False)
            curr_loss = tf.reduce_mean(curr_loss)

            if trainable==1:
                curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_scene_instance')

            if ret_dict==1:
                tmp_loss_list.append(curr_loss)
            else:
                loss_list.append(curr_loss)
            #loss_keys.append('scene_instance')
            now_indx = now_indx + 1
        if ret_dict==1:
            loss_list.append(tf.add_n(tmp_loss_list))
            loss_keys.append('scene')

    if cfg_dataset.get('scannet', 0)==1:
        curr_loss = depth_loss(output[now_indx], args[now_indx + arg_offset], depthloss = 2, depth_norm = depth_norm)

        if trainable==1:
            curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_scannet')

        loss_list.append(curr_loss)
        loss_keys.append('scannet')
        now_indx = now_indx + 1

    if cfg_dataset.get('pbrnet', 0)==1:
        if ret_dict==1:
            tmp_loss_list = []

        if cfg_dataset.get('pbr_normal', 1)==1:
            if nfromd==0:
                curr_loss = normal_loss(output[now_indx], args[now_indx + arg_offset], normalloss = normalloss)
            else:
                curr_loss = normal_loss(output[now_indx], get_n_from_d(args[now_indx + arg_offset]), normalloss = normalloss)
                if cfg_dataset.get('pbr_depth', 1)==1:
                    arg_offset = arg_offset - 1

            if trainable==1:
                if normalloss>=1:
                    curr_loss = curr_loss + 1.2
                curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_pbr_normal')

            if ret_dict==1:
                tmp_loss_list.append(curr_loss)
            else:
                loss_list.append(curr_loss)
            #loss_keys.append('pbr_normal')
            now_indx = now_indx + 1

        if cfg_dataset.get('pbr_depth', 1)==1:
            curr_loss = depth_loss(output[now_indx], args[now_indx + arg_offset], depthloss = depthloss, depth_norm = depth_norm)

            if trainable==1:
                curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_pbr_depth')

            if ret_dict==1:
                tmp_loss_list.append(curr_loss)
            else:
                loss_list.append(curr_loss)
            #loss_keys.append('pbr_depth')
            now_indx = now_indx + 1

        if cfg_dataset.get('pbr_instance', 0)==1:
            curr_loss = get_semantic_loss(curr_predict = output[now_indx], curr_truth = args[now_indx + arg_offset], need_mask = True, mask_range = 40)
            curr_loss = tf.reduce_mean(curr_loss)

            if trainable==1:
                curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_pbr_instance')

            if ret_dict==1:
                tmp_loss_list.append(curr_loss)
            else:
                loss_list.append(curr_loss)
            #loss_keys.append('pbr_instance')
            now_indx = now_indx + 1

        if ret_dict==1:
            loss_list.append(tf.add_n(tmp_loss_list))
            loss_keys.append('pbr')

    if cfg_dataset.get('imagenet', 0)==1:
        curr_loss = get_softmax_loss(curr_label = args[now_indx + arg_offset], curr_output = output[now_indx], label_norm = label_norm, multtime = multtime)

        if trainable==1:
            curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_imagenet')

        loss_list.append(curr_loss)
        loss_keys.append('imagenet')
        now_indx = now_indx + 1

        if extra_feat==1:
            # As we don't need normal and depth here, skip
            now_indx = now_indx + 2
            arg_offset = arg_offset - 2

    if cfg_dataset.get('coco', 0)==1:
        curr_loss = get_semantic_loss(curr_predict = output[now_indx], curr_truth = args[now_indx + arg_offset], need_mask = True, mask_range = 0, less_or_large = 1)
        curr_loss = tf.reduce_mean(curr_loss)

        if trainable==1:
            curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_coco')

        loss_list.append(curr_loss)
        loss_keys.append('coco')
        now_indx = now_indx + 1

    if cfg_dataset.get('place', 0)==1:
        curr_loss = get_softmax_loss(curr_label = args[now_indx + arg_offset], curr_output = output[now_indx], label_norm = label_norm, multtime = multtime)

        if trainable==1:
            curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_place')

        loss_list.append(curr_loss)
        loss_keys.append('place')
        now_indx = now_indx + 1

        if extra_feat==1:
            # As we don't need normal and depth here, skip
            now_indx = now_indx + 2
            arg_offset = arg_offset - 2

    if cfg_dataset.get('kinetics', 0)==1:
        curr_loss = get_softmax_loss(curr_label = args[now_indx + arg_offset], curr_output = output[now_indx], label_norm = label_norm, multtime = multtime)

        if trainable==1:
            curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_kinetics')

        loss_list.append(curr_loss)
        loss_keys.append('kinetics')
        now_indx = now_indx + 1

    if cfg_dataset.get('nyuv2', 0)==1:
        curr_loss = depth_loss(output[now_indx], args[now_indx + arg_offset], depthloss = depthloss, depth_norm = depth_norm)

        if trainable==1:
            curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_nyuv2')

        loss_list.append(curr_loss)
        loss_keys.append('nyuv2')
        now_indx = now_indx + 1

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if len(reg_losses)!=0:
        reg_losses = tf.add_n(reg_losses)
        if print_loss==1:
            reg_losses = tf.Print(reg_losses, [tf.add_n(loss_list)], message = 'Real loss')
        loss_list.append(tf.cast(reg_losses, tf.float32))

    if ret_dict==0:
        return tf.add_n(loss_list)
    else:
        final_dict = {key: value for key, value in zip(loss_keys, loss_list)}
        if combine_dict==1:
            cat_list = []
            non_cat_list = []
            for loss_key in final_dict:
                if loss_key in ['place', 'imagenet', 'kinetics']:
                    cat_list.append(final_dict[loss_key])
                elif loss_key in ['scene', 'pbr', 'coco']:
                    non_cat_list.append(final_dict[loss_key])
            
            new_dict = {'category': tf.add_n(cat_list), 'noncategory': tf.add_n(non_cat_list)}
            final_dict = new_dict

        return final_dict
