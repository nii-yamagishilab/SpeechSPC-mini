# Speech Security and Privacy Compendium - mini

This is the code repository hosting a few projects related to speech security and privacy. 

## Background

This repository is maintained by [Xin Wang](http://tonywangx.github.io/), for [JST PRESTO project](https://tonywangx.github.io/presto.html) *Unified Framework for Speech Privacy Protection and Fake Voice Detection*.

<p align="center">
  <img src="https://tonywangx.github.io/_images/figure_presto.jpg" width="600px" alt="Project Logo"/>
</p>

The decorator `mini` means that this repository is not a Pro collection of speech security and privacy tools. 

XW is a beginner to this field. Please feel free to give suggestions and feedback.


## Projects

### 1. Score-level fusion 

This project is for an Interspeech 2024 paper 
*Revisiting and Improving Scoring Fusion for Spoofing-aware Speaker Verification Using Compositional Data Analysis*


* Summary: for spoofing-aware ASV, how should we fuse the scores from a spoofing countermeasure and a conventional ASV?
    * This work shows how score summation is linked to the compositional data analysis.
    * This work shows how a non-linear score fusion theoretically optimizes a Bayesian decision cost -- better than score summation.

* Code: [proj-01-score-fusion-cda-llr](./proj-01-score-fusion-cda-llr)
    * Please follow `README` in the folder and run the code
    
* Other resources:
    * Proof of the theory related to Bayesian decision cost (link to be inserted)
    * Tutorial notebook on fusion of simulated data (link to be inserted)
    * Tutorial notebook on evaluation metrics (t-EER, cllr, and so on)

Please consider cite the paper if you find the above materials useful
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
| [b3_voice_privacy](./tutorials/b3_voiceprivacy_ch) | readable and executable | tutorials on [speaker anonymization](https://www.voiceprivacychallenge.org/) basic methodss

These tutorials are based on Google Colab also linked in XW's [PytorchNN repository](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/tutorials)


## Acknowledgement

This work is supported by JST PRESTO Grant Number JPMJPR23P9, Japan.

---
©2024, Wang Xin