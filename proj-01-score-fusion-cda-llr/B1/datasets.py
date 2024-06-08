#!/usr/bin/python3
"""
Modules for create datasets for SASV.

Two types of modules are available
1. those for loading waveforms
2. those for loading pre-extracted embeddings
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import random
import numpy as np
import pickle
from pathlib import Path
import speechbrain as sb

import tools.wav_tools as wav_tools
import tools.protocol_tools as protocol_tools
from speechbrain.dataio.dataio import read_audio

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2023, Xin Wang"

#####
# Utils
#####
def get_wav_path(root_dir, setname, filename, flag_flac=False):
    """ path = get_wav_path(root_dir, setname, filename, flag_flac)
    Return full path to the file.
    
    Assume ASVspoof2019 data structure
    |- ASVspoof2019_LA_train/flac/...
    |- ASVspoof2019_LA_dev/flac/...
    |- ASVspoof2019_LA_eval/flac/...
    or 
    |- ASVspoof2019_LA_train/wav/...
    |- ASVspoof2019_LA_dev/wav/...
    |- ASVspoof2019_LA_eval/wav/...
    
    input
    -----
      root_dir: str, path to the root directory of ASVspoof2019 LA
      setname: str, train, dev or eval
      filename: str, name of the file
      flag_flac: bool, whether audio file is in flac or wav

    output
    ------
      filepath: str, full path to the audio file
    """
    if flag_flac:
        # this is for loading flac 
        foldername = 'ASVspoof2019_LA_' + setname
        filepath = Path(root_dir) / foldername / 'flac' / filename
        return filepath.with_suffix('.flac')
    else:
        # this is for loading wav
        foldername = 'ASVspoof2019_LA_' + setname
        filepath = Path(root_dir) / foldername / 'wav' / filename
        return filepath.with_suffix('.wav')
        

def get_wav_data(filepath):
    """ sr, data = get_wav_data(filepath)
    Load wav audio data

    input
    -----
      filepath: str, path to the audio file
    
    output
    ------
      sr: int, sampling rate
      data: np.array, (N, ), audio data
    """
    if filepath.suffix == '.flac':
        # speechbrain uses torchaudio to load flac
        # this requires that flac-related libs
        # https://pytorch.org/audio/0.7.0/backend.html#sox-io-backend
        data = read_audio(str(filepath)).numpy()
        sr = None
    elif filepath.suffix == '.wav':
        # use the in-house tool 
        sr, data = wav_tools.waveReadAsFloat(str(filepath))
        data = np.squeeze(data)
    return sr, data


def pad_data(data, expected_len):
    """data_new = pad_data(data, expected_len)
    Pad or trim input data to a fixed length
    
    input
    -----
      data: np.array, (N, )
      expected_len: int, expected length of the data
                    if it is not a int, expected_len = N
    output
    ------
      data_new: np.array, (expected_len, )
    """
    if type(expected_len) is not int:
        data_new = data
    else:
        if data.shape[0] >= expected_len:
            data_new = data[:expected_len]
        else:
            num_repeats = int(expected_len / data.shape[0]) + 1
            data_new = np.tile(data, (num_repeats))[:expected_len]
    return data_new


def pad_pack_data(data_list, expected_len = None):
    """data_new = pad_pack_data(data_list, expected_len)
    Pad or trim input data in a list to a fixed length
    
    input
    -----
      data_list: list of np.array
      expected_len: int, expected length of the data
                    None, it will be max(data length) in data_list
    output
    ------
      data_new: np.array, (K, expected_len), where K=len(data_list)
    """
    maxlen = max([x.size for x in data_list])
    buflen = maxlen if expected_len is None else expected_len
    
    batch_data = np.zeros([len(data_list), buflen], dtype=data_list[0].dtype)
    for idx, data in enumerate(data_list):
        batch_data[idx] = pad_data(data, buflen)
    return batch_data


########
# Datasets for loading waveforms
########
def create_trn_dataset(hparams):
    """for training set
    """
    setname = 'train'

    # ID, speaker, utterance
    train_dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["path_train_csv"])

    # load original CM protocol and create a dictionary for speaker-utterances
    protocol_pd = protocol_tools.load_csv(hparams['path_train_csv'])
    dic_spk_utt = protocol_tools.protocol_to_spk_utt(protocol_pd)

    # load pre-trained embedding dictonaries if necessary
    # dic[utt] -> embedding vector
    if hparams['flag_use_embedding_input']:
        # speaker meta data
        with open(hparams['path_asv_emb_trn'], 'rb') as fb:
            dic_asv_emb = pickle.load(fb)
        with open(hparams['path_cm_emb_trn'], 'rb') as fb:
            dic_cm_emb = pickle.load(fb)
    else:
        dic_asv_emb, dic_cm_emb = None, None

    # function to sample data
    def __sample_trn_trials(speaker, utterance, hparams,
                            dic_asv_emb, dic_cm_emb, dic_spk_utt):
        # random 
        dice = np.random.randint(low=0, high=2, size=2)
        
        if dice[0] > 0:
            # randomly choose a speaker
            speaker = random.choice(list(dic_spk_utt.keys()))
            # randomly sample two utterances, one for enro, one for test
            enro, test = random.sample(dic_spk_utt[speaker]['bonafide'], 2)
            label_cm = np.array([1])
            label_asv = np.array([1])
            enro_spk = speaker
            test_spk = speaker

        elif dice[1] > 0:
            # nontarget bonafide
            speaker, imposter = random.sample(list(dic_spk_utt.keys()), 2)
            enro = random.choice(dic_spk_utt[speaker]['bonafide'])
            test = random.choice(dic_spk_utt[imposter]['bonafide'])
            label_cm = np.array([1])
            label_asv = np.array([0])
            enro_spk = speaker
            test_spk = imposter

        else:
            speaker = random.choice(list(dic_spk_utt.keys()))
            while len(dic_spk_utt[speaker]['spoof']) == 0:
                speaker = random.choice(list(dic_spk_utt.keys()))
            # target spoofed
            enro = random.choice(dic_spk_utt[speaker]['bonafide'])
            test = random.choice(dic_spk_utt[speaker]['spoof'])
            label_cm = np.array([0])
            label_asv = np.array([0])
            enro_spk = speaker
            test_spk = speaker
        
        if hparams['flag_use_embedding_input']:
            test_asv_emb = dic_asv_emb[test]
            test_cm_emb = dic_cm_emb[test]
            test_data = np.concatenate([test_asv_emb, test_cm_emb])

            # load embedding vectors
            enro_asv_emb = dic_asv_emb[enro]
            # no CM embeddings are extracted for enroll utterances
            enro_cm_emb = dic_cm_emb[enro] if enro in dic_cm_emb \
                          else np.zeros_like(test_cm_emb)
            enro_data = np.concatenate([enro_asv_emb, enro_cm_emb])

        else:
            # load audio data
            enro_path = get_wav_path(hparams['path_audio_root'], setname, enro)
            _, enro_data = get_wav_data(enro_path)
        
            test_path = get_wav_path(hparams['path_audio_root'], setname, test)
            _, test_data = get_wav_data(test_path)
        
            # fix length (following SASVbaseline save_embedding functions)
            enro_data = pad_data(enro_data, hparams['audio_max_len'])
            test_data = pad_data(test_data, hparams['audio_max_len'])

        return enro_data, test_data, 1, label_cm, label_asv, \
            enro, test, enro_spk, test_spk

    # add item 
    train_dataset.add_dynamic_item(
        lambda x, y: __sample_trn_trials(
            x, y, hparams, dic_asv_emb, dic_cm_emb, dic_spk_utt),
        takes=['speaker', 'utterance'],
        provides=['enro_data', 'test_data', 
                  'enro_num', 
                  'label_cm', 'label_asv',
                  'enro_name', 'test_name',
                  'enro_spk', 'test_spk'])
    # set output keys
    train_dataset.set_output_keys(
        ['enro_data', 'test_data', 'enro_num', 'label_cm', 'label_asv',
         'enro_name', 'test_name', 'enro_spk', 'test_spk'])

    return train_dataset


def create_dev_dataset(hparams):
    """ dataset for dev set
    """
    setname = 'dev'
    
    # ID,speaker,utterance,label
    dev_dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["path_dev_csv"]
    )
    

    # load embedding vectors if necessary
    if hparams['flag_use_embedding_input']:
        with open(hparams['path_asv_emb_dev'], 'rb') as fb:
            dic_asv_emb = pickle.load(fb)
        with open(hparams['path_cm_emb_dev'], 'rb') as fb:
            dic_cm_emb = pickle.load(fb)
        with open(hparams['path_asv_spk_enrol_dev'], 'rb') as fb:
            dic_asv_spk_enrol = pickle.load(fb)
        if os.path.isfile(hparams['path_cm_spk_enrol_dev']):
            with open(hparams['path_cm_spk_enrol_dev'], 'rb') as fb:
                dic_cm_spk_enrol = pickle.load(fb)
        else:
            dic_cm_spk_enrol = {}
        # we don't need to load enrollment file list
        spk_enroll = None
    else:
        # we don't need embedding dictionary
        dic_asv_emb, dic_cm_emb = None, None
        dic_asv_spk_enrol, dic_cm_spk_enrol = None, None

        # load list of enroll utterances for each speaker
        spk_enroll = dict()
        for filepath in [hparams['path_dev_enroll_f_list'], 
                         hparams['path_dev_enroll_m_list']]:
            with open(filepath, 'r') as fileptr:
                for line in fileptr:
                    # each line spk enrol_1,enrol_2,enrol_3...
                    spk, *utt = line.rstrip().split()
                    spk_enroll[spk] = utt[0].split(',')
   

    # function to sample data
    def __get_dev_trials(speaker, utterance, label, 
                         hparams, 
                         dic_asv_emb, dic_cm_emb, 
                         dic_asv_spk_enrol, dic_cm_spk_enrol):
        
        test = utterance

        # load data
        if hparams['flag_use_embedding_input']:
            # load embedding vector of test utterance
            test_asv_emb = dic_asv_emb[test]
            test_cm_emb = dic_cm_emb[test]
            test_data = np.concatenate([test_asv_emb, test_cm_emb])

            # load enrollment data vectors
            enro_asv_emb = dic_asv_spk_enrol[speaker]
            if speaker in dic_cm_spk_enrol:
                enro_cm_emb = dic_cm_spk_enrol[speaker]
            else:
                enro_cm_emb = np.zeros_like(test_cm_emb)
            enro_data = np.concatenate([enro_asv_emb, enro_cm_emb])
            enro_file = ''
        else:
            # load audio of test utterance
            test_path = get_wav_path(hparams['path_audio_root'], setname, test)
            _, test_data = get_wav_data(test_path)
            test_data = pad_data(test_data, hparams['audio_max_len'])

            # load enrol audio data

            # enro_data will be a list.
            # The customized PaddedBatch will concate lists of enro_data and 
            # create a batch. 
            # Because the number of enroll utts varies across speakers, we 
            # also need to return the number of enroll utt for this speaker.
            enro_data = []
            for enro in spk_enroll[speaker]:
                epath = get_wav_path(hparams['path_audio_root'], setname, enro)
                _, data = get_wav_data(epath)
                enro_data.append(pad_data(data, hparams['audio_max_len']))
            enro_file = spk_enroll[speaker]

        # load labels
        if label == 'target':
            label_cm = np.array([1])
            label_asv = np.array([1])
            label_num_verbose = np.array([hparams['tar_bonafide_label']])
        else:
            label_asv = np.array([0])
            if label == 'nontarget':
                label_cm = np.array([1])
                label_num_verbose = np.array([hparams['nontar_bonafide_label']])
            else:
                label_cm = np.array([0])
                label_num_verbose = np.array([hparams['spoof_label']])
        
        # we know the enro speaker, but not the test speaker
        enro_spk = speaker
        test_spk = ''
        
        return enro_data, test_data, len(enro_data), \
            label_cm, label_asv, label_num_verbose, \
            enro_file, test, enro_spk, test_spk
        
    dev_dataset.add_dynamic_item(
        lambda x, y, z: __get_dev_trials(
            x, y, z, hparams, dic_asv_emb, dic_cm_emb, 
            dic_asv_spk_enrol, dic_cm_spk_enrol),
        takes=['speaker', 'utterance', 'label'],
        provides=['enro_data', 'test_data', 'enro_num',
                  'label_cm', 'label_asv', 'label_num_verbose',
                  'enro_name', 'test_name', 'enro_spk', 'test_spk'])

    dev_dataset.set_output_keys(
        ['enro_data', 'test_data', 'enro_num', 
         'label_cm', 'label_asv', 'label_num_verbose',
         'enro_name', 'test_name', 'enro_spk', 'test_spk'])

    return dev_dataset


def create_eval_dataset(hparams):
    """ dataset for eval set
    """
    setname = 'eval'
    
    # ID,speaker,utterance,label
    eval_dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["path_eval_csv"]
    )
    
    # load embedding vectors if necessary
    if hparams['flag_use_embedding_input']:
        with open(hparams['path_asv_emb_eval'], 'rb') as fb:
            dic_asv_emb = pickle.load(fb)
        with open(hparams['path_cm_emb_eval'], 'rb') as fb:
            dic_cm_emb = pickle.load(fb)
        with open(hparams['path_asv_spk_enrol_eval'], 'rb') as fb:
            dic_asv_spk_enrol = pickle.load(fb)
        if os.path.isfile(hparams['path_cm_spk_enrol_eval']):
            with open(hparams['path_cm_spk_enrol_eval'], 'rb') as fb:
                dic_cm_spk_enrol = pickle.load(fb)
        else:
            dic_cm_spk_enrol = {}
        # we don't need to load enrollment file list
        spk_enroll = None
    else:
        # we don't need embedding dictionary
        dic_asv_emb, dic_cm_emb = None, None
        dic_asv_spk_enrol, dic_cm_spk_enrol = None, None

        # load list of enroll utterances for each speaker
        spk_enroll = dict()
        for filepath in [hparams['path_eval_enroll_f_list'], 
                         hparams['path_eval_enroll_m_list']]:
            with open(filepath, 'r') as fileptr:
                for line in fileptr:
                    # each line spk enrol_1,enrol_2,enrol_3...
                    spk, *utt = line.rstrip().split()
                    spk_enroll[spk] = utt[0].split(',')
   

    # function to sample data
    def __get_eval_trials(speaker, utterance, label,
                          hparams, 
                          dic_asv_emb, dic_cm_emb, 
                          dic_asv_spk_enrol, dic_cm_spk_enrol,):

        test = utterance

        # load data
        if hparams['flag_use_embedding_input']:
            # load embedding vector of test utterance
            test_asv_emb = dic_asv_emb[test]
            test_cm_emb = dic_cm_emb[test]
            test_data = np.concatenate([test_asv_emb, test_cm_emb])

            # load enrollment data vectors
            enro_asv_emb = dic_asv_spk_enrol[speaker]
            if speaker in dic_cm_spk_enrol:
                enro_cm_emb = dic_cm_spk_enrol[speaker]
            else:
                enro_cm_emb = np.zeros_like(test_cm_emb)
            enro_data = np.concatenate([enro_asv_emb, enro_cm_emb])
            enro_file = ''
        else:
            # load audio of test utterance
            test_path = get_wav_path(hparams['path_audio_root'], setname, test)
            _, test_data = get_wav_data(test_path)
            test_data = pad_data(test_data, hparams['audio_max_len'])

            # load enrol audio data

            # enro_data will be a list.
            # The customized PaddedBatch will concate lists of enro_data and 
            # create a batch. 
            # Because the number of enroll utts varies across speakers, we 
            # also need to return the number of enroll utt for this speaker.
            enro_data = []
            for enro in spk_enroll[speaker]:
                epath = get_wav_path(hparams['path_audio_root'], setname, enro)
                _, data = get_wav_data(epath)
                enro_data.append(pad_data(data, hparams['audio_max_len']))
            enro_file = spk_enroll[speaker]

        # load labels
        if label == 'target':
            label_cm = np.array([1])
            label_asv = np.array([1])
            label_num_verbose = np.array([hparams['tar_bonafide_label']])
        else:
            label_asv = np.array([0])
            if label == 'nontarget':
                label_cm = np.array([1])
                label_num_verbose = np.array([hparams['nontar_bonafide_label']])
            else:
                label_cm = np.array([0])
                label_num_verbose = np.array([hparams['spoof_label']])
        
        # we know the enro speaker, but not the test speaker
        enro_spk = speaker
        test_spk = ''
        
        return enro_data, test_data, len(enro_data), \
            label_cm, label_asv, label_num_verbose, \
            enro_file, test, enro_spk, test_spk
        
    eval_dataset.add_dynamic_item(
        lambda x, y, z: __get_eval_trials(
            x, y, z, hparams, dic_asv_emb, dic_cm_emb, 
            dic_asv_spk_enrol, dic_cm_spk_enrol),
        takes=['speaker', 'utterance', 'label'],
        provides=['enro_data', 'test_data', 'enro_num',
                  'label_cm', 'label_asv', 'label_num_verbose',
                  'enro_name', 'test_name', 'enro_spk', 'test_spk'])

    eval_dataset.set_output_keys(
        ['enro_data', 'test_data', 'enro_num', 
         'label_cm', 'label_asv', 'label_num_verbose',
         'enro_name', 'test_name', 'enro_spk', 'test_spk'])

    return eval_dataset


if __name__ == "__main__":
    print(__doc__)
