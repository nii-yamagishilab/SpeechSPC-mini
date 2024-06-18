# Speech Security and Privacy Compendium - mini

This is the code repository hosting a few projects related to speech security and privacy. 

## Background

This repository is maintained by [Xin Wang](http://tonywangx.github.io/), for [JST PRESTO project](https://tonywangx.github.io/presto.html) *Unified Framework for Speech Privacy Protection and Fake Voice Detection*.

<p align="left">
  <img src="https://tonywangx.github.io/_images/figure_presto.jpg" width="600px" alt="Project Logo"/>
</p>

The decorator `mini` means that this repository is not a Pro collection of speech security and privacy tools. 

XW is a beginner in this field. Please feel free to give suggestions and feedback.


## Projects

### 1. Score-level fusion for spoofing-aware ASV

<p align="left">
  <img src="https://github.com/TonyWangX/TonyWangX.github.io/blob/9c46ee65c8ca0a34f16c926c87661b682aaaba31/code/source/pic/llr_fusion.png?raw=true" width="300px" alt="Project Logo"/>
</p>

This project is for an Interspeech 2024 paper: 

**Revisiting and Improving Scoring Fusion for Spoofing-aware Speaker Verification Using Compositional Data Analysis**

Summary:

* For spoofing-aware ASV, how should we fuse the scores from a spoofing countermeasure and a conventional ASV?
    * This work shows how score summation is linked to the compositional data analysis.
    * This work shows how a non-linear score fusion theoretically optimizes a Bayesian decision cost -- better than score summation.

* Code: [proj-01-score-fusion-cda-llr](./proj-01-score-fusion-cda-llr)
    * Please follow `README` in the folder and run the code
    
* Other resources:
    * Proof of the theory related to Bayesian decision cost on [Arxiv](https://arxiv.org/abs/2406.10836)
    * Tutorial notebook on fusion  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D9YZLkSTwXkZGnZAtLpl-1w9ZG2hUxOY?usp=sharing)
    * Tutorial notebook on evaluation metrics (to be added)

Please consider citing the paper if you find the above materials useful
```bibtex
@inproceedings{wangRevisiting2024,
  title = {Revisiting and {{Improving Scoring Fusion}} for {{Spoofing-aware Speaker Verification Using Compositional Data Analysis}}},
  booktitle = {Proc. {{Interspeech}}},
  author = {Wang, Xin and Kinnunen, Tomi and Kong Aik, Lee and Noe, Paul-Gauthier and Yamagishi, Junichi},
  year = {2024},
  pages = {(accepted)}
}

```

## Tutorials

This repository also hosts tutorial notebooks. Some of them are mentioned in individual projects. Here is a summary of the tutorial notebook shelves:

| Folder | Status | Contents |
| --- | :-- | :-- |
| [b1_neural_vocoder](./tutorials/b1_neural_vocoder) | readable and executable | tutorials on selected neural vocoders
| [b2_anti_spoofing](./tutorials/b2_anti_spoofing) | partially finished | tutorials on [speech anti-spoofing](https://www.asvspoof.org/) 
| [b3_voice_privacy](./tutorials/b3_voiceprivacy_ch) | readable and executable | tutorials on [speaker anonymization](https://www.voiceprivacychallenge.org/) basic methods

These tutorials are based on Google Colab and also linked in XW's [PytorchNN repository](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/tutorials)


## Acknowledgment

This work is supported by JST PRESTO Grant Number JPMJPR23P9, Japan.

---
```bash
Copyright 2024 Wang Xin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```