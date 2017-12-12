from __future__ import division
import os
import numpy as np
import tensorflow as tf
import tabular as tb
import itertools

from scipy.stats import spearmanr
from dldata.metrics.utils import compute_metric_base
from tfutils import base, data, model, optimizer, utils

from utils import post_process_neural_regression_msplit_preprocessed
from dataprovider import NeuralDataProvider
from yolo_tiny_net import YoloTinyNet


class NeuralDataExperiment():
    """
    Defines the neural data testing experiment
    """
    class Config():
        """
        Holds model hyperparams and data information.
        The config class is used to store various hyperparameters and dataset
        information parameters.
        """
        target_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5','conv_6','conv_7','conv_8','conv_9','fc1',
                         'fc2']
        batch_size = 2
        data_path = '/datasets/neural_data/tfrecords_with_meta'
        noise_estimates_path = '/datasets/neural_data/noise_estimates.npy'
        seed = 5
        crop_size = 224
        thres_loss = 1000
        n_epochs = 90
        common_params = {
            'image_size': crop_size,
            'num_classes': 20,
            'batch_size': 1
            }
        net_params = {
            'boxes_per_cell': 2,
            'weight_decay': 0.0005,
            'cell_size': 4,
            'object_scale':1,
            'noobject_scale':1,
            'class_scale':1,
            'coord_scale':1
            }
        #feature_masks = {}
	extraction_step = None
        extraction_targets = [attr[0] for attr in NeuralDataProvider.ATTRIBUTES] + target_layers
        ytn = YoloTinyNet(common_params,net_params,test=True)
        assert NeuralDataProvider.N_VAL % batch_size == 0, \
                ('number of examples not divisible by batch size!')
        val_steps = int(NeuralDataProvider.N_VAL / batch_size)

    def __init__(self):
        self.feature_masks = {}

    def setup_params(self):
        params = {}

        """
        The validation parameters specify the source of the data to compare the model's 
        responses to (NeuralDataProvider), as well as the functions used to aggregate 
        results across batches (agg_func) and in real time (online_agg_func)
        """
        params['validation_params'] = {
            'VAll': {
                'data_params': {
                    # ImageNet data provider arguments
                    'func': NeuralDataProvider,
                    'data_path': self.Config.data_path,
                    'crop_size': self.Config.crop_size,
                    # TFRecords (super class) data provider arguments
                    'file_pattern': '*.tfrecords',
                    'batch_size': self.Config.batch_size,
                    'shuffle': False,
                    'shuffle_seed': self.Config.seed, 
                    'n_threads': 1,
                },
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': self.Config.batch_size,
                    'seed': self.Config.seed,
                    'capacity': self.Config.batch_size * 10,
                    'min_after_dequeue': self.Config.batch_size * 1,
                },
                'targets': {
                    'func': self.return_outputs,
                    'targets': self.Config.extraction_targets,
                },
                'num_steps': self.Config.val_steps,
                'agg_func': self.neural_analysis,
                'online_agg_func': self.online_agg,
            },
            'V6': {
                'data_params': {
                    # ImageNet data provider arguments
                    'func': NeuralDataProvider,
                    'data_path': self.Config.data_path,
                    'crop_size': self.Config.crop_size,
                    # TFRecords (super class) data provider arguments
                    'file_pattern': '*.tfrecords',
                    'batch_size': self.Config.batch_size,
                    'shuffle': False,
                    'shuffle_seed': self.Config.seed, 
                    'n_threads': 1,
                },
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': self.Config.batch_size,
                    'seed': self.Config.seed,
                    'capacity': self.Config.batch_size * 10,
                    'min_after_dequeue': self.Config.batch_size * 1,
                },
                'targets': {
                    'func': self.return_outputs,
                    'targets': self.Config.extraction_targets,
                },
                'num_steps': self.Config.val_steps,
                'agg_func': self.neural_analysisV6,
                'online_agg_func': self.online_agg,
            }
        }

        """
        model_params defines the model to be tested - in our case, YoloTinyNet
        """
        params['model_params'] = {
            'func': self.Config.ytn.validation,
        }

        """
        save_params defines where and when to save the parameters
        """
        params['save_params'] = {
            'host': '35.199.154.71',
            'port': 24444,
            'dbname': 'final',
            'collname': 'yolo',
            'exp_id': 'combined',
            'save_to_gfs': [],
        }

        """
        load_params specifies that the model should be loaded from the collection as saved after training
        """
        params['load_params'] = {
            'host': '35.199.154.71',
            'port': 24444,
            'dbname': 'final',
            'collname': 'yolo',
            'exp_id': 'combined',
            'do_restore': True,
            'load_query': None,
            'query': {'step': self.Config.extraction_step} \
                    if self.Config.extraction_step is not None else None,
        }

        params['inter_op_parallelism_threads'] = 500

        return params


    def return_outputs(self, inputs, outputs, targets, **kwargs):
        """
        Example of extracting desired targets from the model
        """
        retval = {}
        for target in targets:
            retval[target] = outputs[target]
        return retval

    def online_agg(self, agg_res, res, step):
        """
        Appends the value for each key
        """
        #self.feature_masks = {}
	if agg_res is None:
            agg_res = {k: [] for k in res}

            # Generate the feature masks
            for k, v in res.items():
                if k in self.Config.target_layers:
                    num_feats = np.product(v.shape[1:])
                    mask = np.random.RandomState(0).permutation(num_feats)[:1024]
                    self.feature_masks[k] = mask

        for k, v in res.items():
            if 'kernel' in k:
                agg_res[k] = v
            elif k in self.Config.target_layers:
                feats = np.reshape(v, [v.shape[0], -1])
                feats = feats[:, self.feature_masks[k]]
                agg_res[k].append(feats)
            else:
                agg_res[k].append(v)
        return agg_res

    #def online_agg(self, agg_res, res, step):
     #   """
      #  Appends the value for each key
       # """
       # if agg_res is None:
      #      agg_res = {k: [] for k in res}
      #  for k, v in res.items():
      #      if 'kernel' in k:
      #          agg_res[k] = v
      #      else:
      #          agg_res[k].append(v)
      #  return agg_res


    def parse_meta_data(self, results):
        """
        Parses the meta data from tfrecords into a tabarray
        """
        meta_keys = [attr[0] for attr in NeuralDataProvider.ATTRIBUTES                 if attr[0] not in ['images', 'it_feats']]
        meta = {}
        for k in meta_keys:
            if k not in results:
                raise KeyError('Attribute %s not loaded' % k)
            meta[k] = np.concatenate(results[k], axis=0)
        return tb.tabarray(columns=[list(meta[k]) for k in meta_keys], names = meta_keys)


    def categorization_test(self, features, meta, variability=None):
        """
        Performs a categorization test using compute_metric_base from dldata.
        """
        print('Categorization test...')
        if variability is None:
            selection={},
        else:
            selection = {'var': variability}
        category_eval_spec = {
            'npc_train': None,
            'npc_test': 2,
            'num_splits': 20,
            'npc_validate': 0,
            'metric_screen': 'classifier',
            'metric_labels': None,
            'metric_kwargs': {'model_type': 'svm.LinearSVC',
                              'model_kwargs': {'C':5e-3}
                             },
            'labelfunc': 'category',
            'train_q': selection,
            'test_q': selection,
            'split_by': 'obj'
        }
        res = compute_metric_base(features, meta, category_eval_spec)
        res.pop('split_results')
        return res
    
    def regression_test(self, features, IT_features, meta, variability=None):
        """
        Performs a regression test with IT data using compute_metric_base from dldata.
        """
        print('Regression test...')
        if variability is None:
            selection={},
        else:
            selection = {'var': variability}
        it_reg_eval_spec = {
            'npc_train': None,
            'npc_test': 2,
            'num_splits': 20,
            'npc_validate': 0,
            'metric_screen': 'regression',
            'metric_labels': None,
            'metric_kwargs': {'model_type': 'pls.PLSRegression',
                              'model_kwargs': {'n_components':25,'scale':False}
                             },
            'labelfunc': lambda x: (IT_features, None),
            'train_q': selection,
            'test_q': selection,
            'split_by': 'obj'
        }
        res = compute_metric_base(features, meta, it_reg_eval_spec)
        espec = (('all','','IT_regression'), it_reg_eval_spec)
        post_process_neural_regression_msplit_preprocessed(
                res, self.Config.noise_estimates_path)
        res.pop('split_results')
        return res
    
    def compute_rdm(self, features, meta, mean_objects=False):
        """
        Computes the RDM of the input features, creating a 
        [N_CATEGORIES x N_FEATURES] result.
        """
        print('Computing RDM...')
        if mean_objects:
            object_list = list(itertools.chain(
                *[np.unique(meta[meta['category'] == c]['obj']) \
                        for c in np.unique(meta['category'])]))
            features = np.array([features[(meta['obj'] == o.rstrip('_'))].mean(0)                     for o in object_list])
        rdm = 1 - np.corrcoef(features)
        return rdm
    
    def compute_rdmV6(self, features, meta, mean_objects=False):
        """
        Computes the RDM of the input features, creating
        a [N_CATEGORIES x N_FEATURES] result.
        """
        print('Computing RDM V6...')
        if mean_objects:
            object_list = list(itertools.chain(
                *[np.unique(meta[(meta['var'] == 'V6') & (meta['category'] == c)]['obj']) \
                        for c in np.unique(meta['category'])]))
            features = np.array([features[(meta['obj'] == o.rstrip('_'))].mean(0)                     for o in object_list])
        rdm = 1 - np.corrcoef(features)
        return rdm


    def get_features(self, results, num_subsampled_features=None):
        """
        Extracts, preprocesses and subsamples the target features
        and the IT features
        """
        features = {}
        for layer in self.Config.target_layers:
            feats = np.concatenate(results[layer], axis=0)
            feats = np.reshape(feats, [feats.shape[0], -1])
            if num_subsampled_features is not None:
                features[layer] = feats[:, np.random.RandomState(0).permutation(feats.shape[1])[:num_subsampled_features]]

        IT_feats = np.concatenate(results['it_feats'], axis=0)

        return features, IT_feats

    def neural_analysis(self, results):
        """
        Performs an analysis of the results from the model on the neural data.
        This analysis includes:
            - computing a RDM
            - a categorization test
            - and an IT regression.
        """
        #retval = {'conv_1': results['conv_1']}
        retval = {}
        print('Performing neural analysis...')
        meta = self.parse_meta_data(results)
        features, IT_feats = self.get_features(results, num_subsampled_features=1024)

        print('IT:')
        #retval['rdm_it'] = self.compute_rdm(IT_feats, meta, mean_objects=True)

        for layer in features:
            print('Layer: %s' % layer)
            # RDM
            #retval['rdm_%s' % layer] = self.compute_rdm(features[layer], meta, mean_objects=True)
            # RDM correlation
            #retval['spearman_corrcoef_%s' % layer] =                     spearmanr(
            #                np.reshape(retval['rdm_%s' % layer], [-1]),
            #                np.reshape(retval['rdm_it'], [-1])
            #                )[0]
            # categorization test
            #retval['categorization_%s' % layer] = self.categorization_test(features[layer], meta, ['V0','V3','V6'])
            # IT regression test
            retval['it_regression_%s' % layer] = self.regression_test(features[layer], IT_feats, meta, ['V0','V3','V6'])
        return retval
    
    def neural_analysisV6(self, results):
        """
        Performs an analysis of the results from the model on the neural data.
        This analysis includes:
            - computing a RDM
            - a categorization test
            - and an IT regression.
        """
        #retval = {'conv_1': results['conv_1']}
        retval = {}
        print('Performing neural analysis...')
        meta = self.parse_meta_data(results)
        features, IT_feats = self.get_features(results, num_subsampled_features=1024)

        print('IT:')
        #retval['rdm_it'] = self.compute_rdmV6(IT_feats, meta, mean_objects=True)

        for layer in features:
            print('Layer: %s' % layer)
            # RDM
            #retval['rdm_%s' % layer] = self.compute_rdmV6(features[layer], meta, mean_objects=True)
            # RDM correlation
            #retval['spearman_corrcoef_%s' % layer] = spearmanr(
            #                np.reshape(retval['rdm_%s' % layer], [-1]),
            #                np.reshape(retval['rdm_it'], [-1])
            #                )[0]
            # categorization test
            #retval['categorization_%s' % layer] = self.categorization_test(features[layer], meta, ['V6'])
            # IT regression test
            retval['it_regression_%s' % layer] = self.regression_test(features[layer], IT_feats, meta, ['V6'])
                
        return retval

if __name__ == '__main__':
    """
    Illustrates how to run the configured model using tfutils
    """
    base.get_params()
    m = NeuralDataExperiment()
    params = m.setup_params()
    base.test_from_params(**params)

