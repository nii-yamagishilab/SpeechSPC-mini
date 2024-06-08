#!/usr/bin/python3
"""
Definition of model
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import random
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb

from tqdm.contrib import tqdm

import sklearn
import sklearn.linear_model
from sklearn.mixture import GaussianMixture

from tools import metric_sasv
from tools import io_tools
from tools import util_torch

from submodules.ecapa_tdnn import model as asv_model
from submodules.aasist import AASIST as cm_model


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2023, Xin Wang"


def fit_gaussian(data):
    # fit gaussian
    gm = GaussianMixture(n_components=1, random_state=0)
    gm.fit(data)
    
    # get mean, precision matrix, and determinant
    m = gm.means_[0]
    cov = gm.covariances_[0]
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    return m, inv, det

def logistic_reg(input_data, labels, pos_prior):
    """
    """
    prior_weight = {0: 1-pos_prior, 1:pos_prior}
    prior_logit = np.log(pos_prior /(1-pos_prior))

    reg_model = sklearn.linear_model.LogisticRegression(
        class_weight = prior_weight)

    if input_data.ndim == 1:
        reg_model.fit(np.expand_dims(input_data, axis=1), labels)
    else:
        reg_model.fit(input_data, labels)

    scale = torch.tensor(reg_model.coef_[0]) 
    bias = torch.tensor(reg_model.intercept_) - prior_logit
    return scale, bias
        

class LogSoftmaxWrapperWeighted(torch.nn.Module):
    """
    """
    def __init__(self, loss_fn, weight):
        super(LogSoftmaxWrapperWeighted, self).__init__()
        self.loss_fn = loss_fn
        self.criterion = torch.nn.KLDivLoss(reduction="none")
        self.weight = torch.nn.parameter.Parameter(weight, 
                                                   requires_grad=False)

    def forward(self, outputs, targets, length=None):
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        targets = F.one_hot(targets.long(), outputs.shape[1]).float()
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)

        predictions = F.log_softmax(predictions, dim=1)
        loss_mat = self.criterion(predictions, targets) 
        weight_mat = targets * self.weight
        loss = (loss_mat * weight_mat).sum() / weight_mat.sum()
        return loss 


class FrontEnd(torch.nn.Module):
    """
    """
    def __init__(self, config):
        super(FrontEnd, self).__init__()

        self.asv_emb_dim = config['asv_config']['emb_dim']
        self.cm_emb_dim = config['cm_config']['emb_dim']

        # pre-trained path
        self.asv_path = config['asv_pretrn'] if 'asv_pretrn' in config else None
        self.cm_path = config['cm_pretrn'] if 'cm_pretrn' in config else None

        # update front end?
        self.flag_update = config['update']
                
        # use flag
        self.flag_use_embedding_input = config['frontend_use_embedding_input']
        if self.flag_use_embedding_input:
            # use pre-extracted embeddings
            self.m_asv_model = None
            self.m_cm_model = None
        else:
            # use models to get embeddings
            # asv model
            self.m_asv_model = asv_model.ECAPA_TDNN(config['asv_config']['ch'])
            # cm model
            self.m_cm_model = cm_model.Model(config['cm_config'])
            # initialize the models
            self._init_models()

        # loading enrollment embedding tasks time. Create a dictionary to cache
        # (we can do the same for all embeddings, but currently only for enrol)
        self.cache_enrol_cm = dict()
        self.cache_enrol_asv = dict()

        return

    def _init_models(self):
        # load pre-trained models if necessary
        if self.asv_path:
            util_torch.load_weights(self.m_asv_model.state_dict(),self.asv_path)
        if self.cm_path:
            util_torch.load_weights(self.m_cm_model.state_dict(), self.cm_path)
        return

    def _reshape_before(self, audio_data, data_length):
        batch_size = len(data_length)
        mat_len = audio_data.shape[1]

        num_test = 1
        num_enro = audio_data.shape[-1] - num_test
        num_test = num_test * batch_size
        num_enro = num_enro * batch_size

        # both enroll and test audio data are included, reshape it to 
        # the batch dimension
        # (batch, length, enro + test) -> ((enro + test) * batch , length)
        audio_data_ = torch.permute(audio_data, (2, 0, 1)).contiguous()
        audio_data_ = torch.reshape(audio_data_, (-1, mat_len))
        return audio_data_, batch_size, num_test, num_enro
        
    def _reshape_after(self, asv_emb, cm_emb, batch_size, num_test, num_enro):
        # enroll data first, test data later
        def _reshape_after_core(input_tensor, batch_size, num_test, num_enro):
            return input_tensor[:num_enro], input_tensor[num_enro:]
        asv_emb_enro, asv_emb_test = _reshape_after_core(
            asv_emb, batch_size, num_test, num_enro)
        cm_emb_enro, cm_emb_test = _reshape_after_core(
            cm_emb, batch_size, num_test, num_enro)
        return asv_emb_enro, asv_emb_test, cm_emb_enro, cm_emb_test
        

    def _merge_enrol_emb(self, enrol_emb, enro_num):
        if any(enro_num > 1):
            output = torch.zeros_like(enrol_emb[:len(enro_num)])
            s_idx = 0
            for idx, num in enumerate(enro_num):
                output[idx] = enrol_emb[s_idx: s_idx + int(num)].mean(dim=0)
                s_idx += int(num)
        else:
            output = enrol_emb
        return output

    def _forward_audio(self, 
                       enro_data, enro_length, 
                       test_data, test_length, 
                       enro_num, enro_spk):
        
        if self.flag_update:
            # if front end is to be updated 

            # asv embedding
            enro_asv_emb = self.m_asv_model(enro_data)            
            asv_emb = self.m_asv_model(test_data)

            # cm embedding
            enro_cm_emb = self.m_cm_model(enro_data)
            cm_emb, _ = self.m_cm_model(test_data)

        else:
            # if front end is frozen
            
            self.m_asv_model.eval()
            self.m_cm_model.eval()
            with torch.no_grad():

                # embeddings for test utterance
                asv_emb = self.m_asv_model(test_data)
                cm_emb, _ = self.m_cm_model(test_data)

                # embeddings for enroll utterances
                if sum(enro_num) > len(enro_num):

                    # in case of multiple enroll utts per test trial,
                    # the total number of enroll utts may be huge.
                    # process the enroll for each test trial separately
                    enro_asv_emb = torch.zeros_like(asv_emb)
                    enro_cm_emb = torch.zeros_like(cm_emb)
                    
                    s_idx = 0
                    for idx, (spk, num) in enumerate(zip(enro_spk, enro_num)):
                        num = num.item()
                        # for the idx data, its enroll spk is spk
                        if spk in self.cache_enrol_cm \
                           and spk in self.cache_enrol_asv:
                            # if available in cache
                            enro_asv_emb[idx] = self.cache_enrol_asv[spk]
                            enro_cm_emb[idx] = self.cache_enrol_cm[spk]
                        else:
                            # get enroll embeddings and 
                            # compute average over enroll embeddings
                            enro_asv_emb[idx] = self.m_asv_model(
                                enro_data[s_idx: s_idx + num]).mean(dim=0)
                            enro_cm_emb[idx] = self.m_cm_model(
                                enro_data[s_idx: s_idx + num])[0].mean(dim=0)
                            # save to cache as well
                            self.cache_enrol_asv[spk] = enro_asv_emb[idx]
                            self.cache_enrol_cm[spk] = enro_cm_emb[idx]

                        # move the pointer
                        s_idx += num
                    
                else:
                    # each data has one enrollment
                    enro_asv_emb = self.m_asv_model(enro_data)            
                    enro_cm_emb, _ = self.m_cm_model(enro_data)
                
        return enro_asv_emb, asv_emb, enro_cm_emb, cm_emb


    def _forward_emb(self, enro_data, test_data):
        asv_emb = test_data[:, :self.asv_emb_dim]
        cm_emb = test_data[:,  self.asv_emb_dim:]
        enro_asv_emb = enro_data[:, :self.asv_emb_dim]
        enro_cm_emb = enro_data[:, self.asv_emb_dim:]
        if enro_data.shape[-1] == self.asv_emb_dim:
            enro_cm_emb = torch.zeros_like(cm_emb)
        return enro_asv_emb, asv_emb, enro_cm_emb, cm_emb
    
    def forward(self, enro_data, enro_length, test_data, test_length, 
                enro_num, enro_spk):
        if self.flag_use_embedding_input:
            return self._forward_emb(enro_data, test_data)
        else:
            return self._forward_audio(enro_data, enro_length, 
                                       test_data, test_length, 
                                       enro_num, enro_spk)


class BackEnd(torch.nn.Module):
    """
    """
    def __init__(self, config):
        super(BackEnd, self).__init__()
        
        # embedding size
        self.cm_emb_dim = config['cm_emb_dim']
        self.asv_emb_dim = config['asv_emb_dim']
        
        # CM scoring backend
        # number of hidden nodes
        l_nodes = [256, 128, 64] 
        l_fc = []
        for idx in range(len(l_nodes)):
            if idx == 0:
                l_fc.append(torch.nn.Linear(
                    in_features = self.cm_emb_dim,
                    out_features = l_nodes[idx]))
            else:
                l_fc.append(torch.nn.Linear(
                    in_features = l_nodes[idx-1],
                    out_features = l_nodes[idx]))
            l_fc.append(torch.nn.LeakyReLU(negative_slope = 0.3))
        l_fc.append(torch.nn.Linear(l_nodes[-1], 2, bias = False))
        self.m_classifier = torch.nn.Sequential(*l_fc)

            
        # ASV scoring method
        self.m_cossim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        
        # Parameters for the three Gaussians
        # target bona fide
        self.m_mean_tb = torch.nn.Parameter(
            torch.zeros([2]), requires_grad=False)
        self.m_pre_tb = torch.nn.Parameter(
            torch.ones([2, 2]), requires_grad=False)
        self.m_det_tb = torch.nn.Parameter(
            torch.ones([1]), requires_grad=False)
        
        # non-target bona fide
        self.m_mean_nb = torch.nn.Parameter(
            torch.zeros([2]), requires_grad=False)
        self.m_pre_nb = torch.nn.Parameter(
            torch.ones([2, 2]), requires_grad=False)
        self.m_det_nb = torch.nn.Parameter(
            torch.ones([1]), requires_grad=False)
        
        # spoofed
        self.m_mean_sf = torch.nn.Parameter(
            torch.zeros([2]), requires_grad=False)
        self.m_pre_sf = torch.nn.Parameter(
            torch.ones([2, 2]), requires_grad=False)
        self.m_det_sf = torch.nn.Parameter(
            torch.ones([1]), requires_grad=False)

        # parameters for calibration of Gaussian LLRs
        self.m_reg_cm_a = torch.nn.Parameter(
            torch.ones([1]), requires_grad=False)
        self.m_reg_cm_b = torch.nn.Parameter(
            torch.zeros([1]), requires_grad=False)        
        self.m_reg_asv_a = torch.nn.Parameter(
            torch.ones([1]), requires_grad=False)
        self.m_reg_asv_b = torch.nn.Parameter(
            torch.zeros([1]), requires_grad=False)

        # prior parameters (alpha)
        self.m_alpha = torch.nn.Parameter(
            torch.ones([1]), requires_grad=False)

        return

    def forward(self, asv_enro, asv_test, cm_enro, cm_test):
        """
        """
        
        # ===
        # get the raw scores
        # ===
        # for ASV part, we simply measure cosine
        asv_cossim = self.m_cossim(asv_enro, asv_test)
        #asv_logits = (asv_cossim + 1.0) / 2.0
        #asv_logits = torch.log(asv_logits / (1 - asv_logits))
        asv_logits = asv_cossim
        asv_scores = self._get_asv_scores(asv_logits)
        
        # for CM part
        # cm_logits -> (batch, num_output_class)
        cm_logits = self.m_classifier(cm_test)
        cm_scores = self._get_cm_scores(cm_logits)

        # ===
        # get Gaussian log-likelihood ratios 
        # ===
        # get LLRs for CM and ASV
        llr_tarspf, llr_tarnon = self._gaussian_llrs(cm_scores, asv_scores)

        # calibration
        llr_tarspf = self._get_cm_calib(llr_tarspf)
        llr_tarnon = self._get_asv_calib(llr_tarnon)

        # ===
        # Gaussian fusion
        # ===
        # irl
        sasv_s = self._gaussian_scoring(llr_tarspf, llr_tarnon, self.m_alpha)

        return cm_logits, llr_tarspf, asv_logits, llr_tarnon, sasv_s    

    def _create_gaussian_data(self, cm_scores, asv_scores):
        return torch.stack([cm_scores, asv_scores], dim=1)

    def _gaussian_scoring(self, llr_tarspf, llr_tarnon, alpha):
        """sasv_score = _gaussian_scoring(llr_tarspf, llr_tarnon, alpha)
        SASV score fusion based on the three Gaussian models
        """
        # log sum exp trick to prevent overflow
        # it does not affect the result
        shift = (llr_tarspf + llr_tarnon) / 2.0

        # weighted sum of the LRs
        tmp1 = alpha * torch.exp(-llr_tarspf + shift) 
        tmp2 = (1-alpha) * torch.exp(-llr_tarnon + shift)

        # remember to take the min_val out
        return - (torch.log(tmp1 + tmp2) - shift)
        
    def _gaussian_calib_fit(
            self, 
            cm_scores, asv_scores, labels, 
            tarbon_label_val, nonbon_label_val, spoof_label_val):
        """
        Fit the three Gaussians and calibration parameters using dev data
        """
        # compose the data
        data_ = self._create_gaussian_data(cm_scores, asv_scores)

        # fitting is done using Numpy
        data = data_.detach().cpu().numpy()
        
        # get the labels (for SASV)
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()  

        # get the labels for CM (
        # CM label: use all data tar+non versus spoof
        index_cm_data = np.full(labels.shape, True)
        # [True: bona fide, False: spoofed]
        labels_cm = labels != spoof_label_val

        # ASV label: only consider target bona and nontarget bona
        index_asv_data = labels != spoof_label_val
        # [True: target, False: nontarget]
        labels_asv = labels[index_asv_data] == tarbon_label_val
        
        # priors for logistic reg
        dev_cm_prior = np.sum(labels_cm) / labels_cm.size
        dev_asv_prior = np.sum(labels_asv) / labels_asv.size
            
        # ===
        # fit the three gaussians
        # ===
        data_tar_bon = data[labels == tarbon_label_val]
        m, inv, det = fit_gaussian(data_tar_bon)
        self.m_mean_tb.copy_(torch.from_numpy(m))
        self.m_pre_tb.copy_(torch.from_numpy(inv))
        self.m_det_tb.copy_(torch.tensor(det))

        data_non_bon = data[labels == nonbon_label_val]
        m, inv, det = fit_gaussian(data_non_bon)
        self.m_mean_nb.copy_(torch.from_numpy(m))
        self.m_pre_nb.copy_(torch.from_numpy(inv))
        self.m_det_nb.copy_(torch.tensor(det))

        data_spoof = data[labels == spoof_label_val]
        m, inv, det = fit_gaussian(data_spoof)
        self.m_mean_sf.copy_(torch.from_numpy(m))
        self.m_pre_sf.copy_(torch.from_numpy(inv))
        self.m_det_sf.copy_(torch.tensor(det))
        
        # ===
        # get the LLRs
        # ===
        # on torch, compute LLRs
        llr_tarspf, llr_tarnon = self._gaussian_llrs(cm_scores, asv_scores)
        
        # ===
        # estimate parameters for calibration
        # ===
        # regression for CM
        cm_a, cm_b = logistic_reg(
            llr_tarspf.detach().cpu().numpy()[index_cm_data], 
            labels_cm, dev_cm_prior)

        asv_a, asv_b = logistic_reg(
            llr_tarnon.detach().cpu().numpy()[index_asv_data],
            labels_asv, dev_asv_prior)

        # save the coef back to model
        self.m_reg_cm_a.copy_(cm_a)
        self.m_reg_cm_b.copy_(cm_b)
        self.m_reg_asv_a.copy_(asv_a)
        self.m_reg_asv_b.copy_(asv_b)


        # ===
        # grid search for alpha
        # ===
        # get the calibrated llrs
        llr_tarspf_calib = self._get_cm_calib(llr_tarspf)
        llr_tarnon_calib = self._get_asv_calib(llr_tarnon)

        # grid search for alpha
        eer = 1.0
        alpha_best = None
        step_num = 10
        iter_num = 5
        grid_start = 0.0
        grid_end = 1.0
        
        for iter_idx in np.arange(iter_num):
            # grid search
            for alpha in np.linspace(grid_start, grid_end, step_num):
                scores = self._gaussian_scoring(
                    llr_tarspf_calib, llr_tarnon_calib, alpha)
                
                eer_tmp, _, _, _, _, _ = metric_sasv.compute_sasv_eer(
                    scores.detach().cpu().numpy(), labels)
                
                if eer_tmp <= eer:
                    eer = eer_tmp
                    alpha_best = alpha

            # reduce the search range
            if (grid_end - alpha_best) < (alpha_best - grid_start):
                # alpha_best is larger than the mid point
                grid_start = alpha_best
            else:
                grid_end = alpha_best
            
            # if the range is 0, no need to search anymore
            if grid_end == grid_start:
                break
            
        # save alpha and beta
        self.m_alpha.copy_(alpha_best)
        
        # done
        return


    def _gaussian_ll(self, data, mean, pre, det):
        """ 
        compute log-likelihood given data, mean, precision matrix, and 
        determant of covariance matrix
        """
        data_centered = data - mean
        ll = (torch.mm(data_centered, pre) * data_centered).sum(dim=1)
        ll = - np.log(2 * np.pi) - 0.5 * torch.log(det) - 0.5 * ll
        return ll

    def _gaussian_tb_ll(self, data):
        # ll of tar-bonafide Gaussian
        return self._gaussian_ll(
            data, self.m_mean_tb, self.m_pre_tb, self.m_det_tb)

    def _gaussian_nb_ll(self, data):
        # ll of non-target-bonafide Gaussian
        return self._gaussian_ll(
            data, self.m_mean_nb, self.m_pre_nb, self.m_det_nb)

    def _gaussian_sf_ll(self, data):
        # ll of spoof Gaussian
        return self._gaussian_ll(
            data, self.m_mean_sf, self.m_pre_sf, self.m_det_sf)

    def _gaussian_llrs(self, cm_scores, asv_scores):
        """
        """
        # compose 2D data
        data = self._create_gaussian_data(cm_scores, asv_scores)
        
        # log-likelihood of tar bona gaussian
        ll_tb = self._gaussian_tb_ll(data)
        # log-likelihood of nontar bona gaussian
        ll_nb = self._gaussian_nb_ll(data)
        # log-likelihood of spoof gaussian
        ll_sf = self._gaussian_sf_ll(data)

        # for CM
        llr_tarspf = ll_tb - ll_sf
        # for ASV
        llr_tarnon = ll_tb - ll_nb

        return llr_tarspf, llr_tarnon

    def _get_cm_scores(self, cm_logits):
        # convert CM logits into CM raw score
        cm_scores = cm_logits[:, 1] - cm_logits[:, 0]
        return cm_scores

    def _get_asv_scores(self, asv_logits):
        # convert ASV logits to ASV score
        # here, no need to do anything
        return asv_logits

    def _get_cm_calib(self, llr_tarspf):
        # calibration of CM LLRs 
        return self.m_reg_cm_a * llr_tarspf + self.m_reg_cm_b

    def _get_asv_calib(self, llr_tarnon):
        # calibration of ASV LLRs
        return self.m_reg_asv_a * llr_tarnon + self.m_reg_asv_b

    def get_raw_cm_score(self, cm_logits, llr_tarspf, asv_logits, llr_tarnon, 
                         tmp):
        """API to return output of CM, given the output from
        Backend -> forward()
        """
        return self._get_cm_scores(cm_logits)
    
    def get_raw_asv_score(self, cm_logits, llr_tarspf, asv_logits, llr_tarnon,
                          tmp):
        """API to return output of ASV, given the output from
        Backend -> forward()
        """
        return self._get_asv_scores(asv_logits)

    
    def get_score(self, cm_logits, llr_tarspf, asv_logits, llr_tarnon, sasv_s):
        """API to return the score used for SASV, given the output from
        Backend -> forward()
        """
        return sasv_s

    def get_cm_scores(self, cm_logits, llr_tarspf, asv_logits, llr_tarnon, tmp):
        """API to return the score for CM, given the output from
        Backend -> forward()
        """
        # this llrs has been calibrated in forward() by calibration function
        return llr_tarspf
    
    def get_asv_scores(self, cm_logits, llr_tarspf, asv_logits, llr_tarnon, tmp):
        """API to return the score for ASV, given the output from
        Backend -> forward()
        """
        # this llrs has been calibrated in forward() by calibration function
        return llr_tarnon
       

class BackEndLoss(torch.nn.Module):
    def __init__(self):
        super(BackEndLoss, self).__init__()

        self.ce = LogSoftmaxWrapperWeighted(
            torch.nn.Identity(), torch.tensor([0.1, 0.9]))
        self.bce = torch.nn.BCEWithLogitsLoss()
        return
    
    def forward(self, cm_logits, llr_tarspf, asv_logits, llr_tarnon, 
                sasv_score, label_cm, label_asv):

        # BCE loss for CM
        loss_cm = self.ce(cm_logits, label_cm) 
        
        # we only tune the CM.
        return loss_cm

        
class SASVBaseline(sb.core.Brain):
    """
    """
    def __init__(self, *args, **kargs):
        """
        """
        # initialization of the parent class
        super(SASVBaseline, self).__init__(*args, **kargs)
        
        # checkpointer will be initialized from yaml
        # if case the model is not defined in yaml, add them here
        # to the checkpointer recoverable settings
        self.checkpointer.add_recoverables(
            {
                'frontend': self.modules.frontend,
                'backend': self.modules.backend,
                'counter': self.hparams.epoch_counter
            }
        )

        return

    def load_checkpoint(self):
        """load_best_ckpt()
        
        Load the best checkpoint from save directory specified in 
        checkpointer.checkpoints_dir
        """
        # find the ckpt
        best_ckpt = self.checkpointer.find_checkpoint(min_key="sasv_eer")
        
        # load models (back and front ends)
        best_paramfile = best_ckpt.paramfiles["backend"]
        sb.utils.checkpoints.torch_parameter_transfer(
            self.modules.backend, best_paramfile, device=self.device)

        best_paramfile = best_ckpt.paramfiles["frontend"]
        sb.utils.checkpoints.torch_parameter_transfer(
            self.modules.frontend, best_paramfile, device=self.device)
        
        return

    @staticmethod
    def create_modules(config):
        """create_modules(config)

        Speechbrain defines model in yaml. When initializing a sb.score.Brain 
        instance (a model), we can use this method to return modules required
        by sb.score.Brain. By doing this, we can define the model in *.py,
        not in yaml.
        """
        return {'frontend': FrontEnd(config['frontend_config']),
                'backend': BackEnd(config['backend_config']), 
                'loss_compute': BackEndLoss()}

    def compute_forward(self, batch, stage):
        """
        """
        batch = batch.to(self.device)
        enro_audio_data, enro_data_length = batch.enro_data
        test_audio_data, test_data_length = batch.test_data
        enro_num = batch.enro_num
        enro_spk = batch.enro_spk

        # forward computation
        # front end
        asv_emb_enro, asv_emb_test, cm_emb_enro, cm_emb_test \
            = self.modules.frontend(
                enro_audio_data, enro_data_length,
                test_audio_data, test_data_length,
                enro_num, enro_spk)

        # back end
        output = self.modules.backend(
            asv_emb_enro, asv_emb_test, cm_emb_enro, cm_emb_test)
        
        return output

    def compute_objectives(self, backend_output, batch, stage):
        """
        """
        # get the labels
        label_cm, _ = batch.label_cm
        label_asv, _ = batch.label_asv

        # compute the loss given the labels
        loss = self.modules.loss_compute(*backend_output, label_cm, label_asv)

        if stage != sb.Stage.TRAIN:
            
            # save all the raw output from CM and ASV
            # they will be used in on_stage_end to git Gaussian and calibration
            tmp = torch.stack(
                [
                    self.modules.backend.get_raw_cm_score(*backend_output),
                    self.modules.backend.get_raw_asv_score(*backend_output)
                ],dim=1)
            self.dev_score_bag.append(tmp)
            
            # save all the labels
            label_num_verbose, _ = batch.label_num_verbose
            self.dev_label_bag.append(label_num_verbose)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            # clean the buffer whenever the new validation around begins
            self.dev_score_bag = []
            self.dev_label_bag = []
    
    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID:

            # raw ASV and CM scores
            tmp = torch.concatenate(self.dev_score_bag, dim=0)
            dev_cm_scores = tmp[:, -2]
            dev_asv_scores = tmp[:, -1]

            # labels
            dev_labels = torch.concatenate(self.dev_label_bag).squeeze(-1)
            dev_labels = dev_labels.cpu().numpy()

            # fit gaussians and get the calibration parameters
            self.modules.backend._gaussian_calib_fit(
                dev_cm_scores, dev_asv_scores,
                dev_labels,
                self.hparams.tar_bonafide_label,
                self.hparams.nontar_bonafide_label,
                self.hparams.spoof_label
            )

            # use conventional approach as baseline for validation
            dev_scores = dev_cm_scores + dev_asv_scores

            # 
            sasv_eer, sv_eer, cm_eer, _, _, _ = metric_sasv.compute_sasv_eer(
                dev_scores.detach().cpu().numpy(), dev_labels)

            print('SASVEER: {:.2f}, ASVEER: {:.2f}, CMEER: {:.2f}'.format(
                sasv_eer * 100, sv_eer * 100, cm_eer * 100))


            stage_stats = {'sasv_eer': sasv_eer.item(),
                           'sv_eer': sv_eer.item(),
                           'cm_eer': cm_eer.item()}            

            self.hparams.train_logger.log_stats(
	        stats_meta={"epoch": epoch},
                train_stats=None,
                valid_stats=stage_stats,
            )
            
            self.checkpointer.save_checkpoint(meta=stage_stats)
            #self.checkpointer.save_and_keep_only(
            #    meta=stage_stats, min_keys=["sasv_eer"])

        return            

    def test(self, eval_dataset, eval_loader_para):
        
        eval_dataloader = sb.dataio.dataloader.make_dataloader(
            eval_dataset, **eval_loader_para)
        
        score_bag = []
        label_bag = []
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, dynamic_ncols=True): 

                batch = batch.to(self.device)
                outputs = self.compute_forward(batch, None)
                label_num_verbose, _ = batch.label_num_verbose
                
                score_bag.append(self.modules.backend.get_score(*outputs))
                label_bag.append(label_num_verbose)
                
            scores = torch.concatenate(score_bag)
            # 
            labels = torch.concatenate(label_bag).squeeze(-1)
            # 
            sasv_eer, sv_eer, cm_eer, _, _, _ = metric_sasv.compute_sasv_eer(
                scores.detach().cpu().numpy(),
                labels.detach().cpu().numpy())
            print('SASVEER: {:.2f}, ASVEER: {:.2f}, CMEER: {:.2f}'.format(
                sasv_eer * 100, sv_eer * 100, cm_eer * 100))

            stage_stats = {'sasv_eer': sasv_eer.item(),
                           'sv_eer': sv_eer.item(),
                           'cm_eer': cm_eer.item()}            

            self.hparams.eval_logger.log_stats(
	        stats_meta={},
                train_stats=None,
                valid_stats=None,
                test_stats=stage_stats
            )

        return


    def analysis(self, dev_dataset, dev_loader_para, save_folder):

        dev_dataloader = sb.dataio.dataloader.make_dataloader(
            dev_dataset, **dev_loader_para)
        
        score_bag = []
        label_bag = []
        cm_score_bag = []
        asv_score_bag = []
        with torch.no_grad():
            for batch in tqdm(dev_dataloader, dynamic_ncols=True): 

                batch = batch.to(self.device)
                outputs = self.compute_forward(batch, None)
                label_num_verbose, _ = batch.label_num_verbose
                
                score_bag.append(self.modules.backend.get_score(*outputs))
                label_bag.append(label_num_verbose)

                cm_score_bag.append(
                    self.modules.backend.get_cm_scores(*outputs))

                asv_score_bag.append(
                    self.modules.backend.get_asv_scores(*outputs))
                
            # score and labels
            scores = torch.concatenate(score_bag)
            labels = torch.concatenate(label_bag).squeeze(-1)
            cm_scores = torch.concatenate(cm_score_bag)
            asv_scores = torch.concatenate(asv_score_bag)

            sasv_eer, sv_eer, cm_eer, _, _, _ = metric_sasv.compute_sasv_eer(
                scores.detach().cpu().numpy(),
                labels.detach().cpu().numpy())
            print('SASVEER: {:.2f}, ASVEER: {:.2f}, CMEER: {:.2f}'.format(
                sasv_eer * 100, sv_eer * 100, cm_eer * 100))

            # 
            io_tools.pickle_dump(scores.cpu().numpy(), 
                                 save_folder + '/scores.pkl')
            io_tools.pickle_dump(labels.cpu().numpy(), 
                                 save_folder + '/labels.pkl')
            io_tools.pickle_dump(cm_scores.cpu().numpy(), 
                                 save_folder + '/cm_scores.pkl')
            io_tools.pickle_dump(asv_scores.cpu().numpy(), 
                                 save_folder + '/asv_scores.pkl')

        return
        

if __name__ == "__main__":
    print(__doc__)
