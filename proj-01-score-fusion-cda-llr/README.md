
# Score-level fusion for Spoofing-aware ASV


<p align="center">
  <img src="https://github.com/TonyWangX/TonyWangX.github.io/blob/9c46ee65c8ca0a34f16c926c87661b682aaaba31/code/source/pic/llr_fusion.png?raw=true" width="300px" align="left" alt="Project Logo"/>
</p>

This is the code repository for the following paper

```
Xin Wang, Tomi Kinnunen, Lee Kong Aik, Paul-Gauthier Noe, and Junichi Yamagishi. 2024. Revisiting and Improving Scoring Fusion for Spoofing-aware Speaker Verification Using Compositional Data Analysis. In Proc. Interspeech, page (accepted).
```

## Resources

* Appendix on the proof of the theory related to Bayesian decision cost can be found [here](https://www.dropbox.com/scl/fo/ykbnw4t8u09vbl9zyir4l/APrQUbOPPIQSHGpnLtdem3o/misc/is2024-sasv.pdf?rlkey=1at87m1q157rlcx4jo933pxeb)
  * This appendix explains how the optimal score fusion w.r.t a specific Bayesian decision cost is derived.
  * It is based on my beginner's point of view on the theory. It should be easy to digest.
* Tutorial demo on fusion of simulated data can be found here (to be added)
* Tutorial on evaluation metrics (t-EER, cllr, and so on) can be found here (to be added)

## Python environment

1. [Speechbrain](https://github.com/speechbrain/speechbrain)
2. [scikit-learn](https://scikit-learn.org/)

Please follow their official website to install.  Their latest stable versions should work. 

The ones I used are
```bash
Speechbrain: github commit 16ef03604b187ff1d926368963bfda09515a47f0

scikit-learn: 1.3.2
```

## Quick start


**Step1.** Download the protocol, file lists, and so on 

```bash
bash 01_download.sh 
```

**Step2.** Choose a system, train the system
```bash
bash 02_train.sh <system_name>

# for example
bash 02_train.sh B1
```
**Step3.** Scoring the dev and eval sets, and get the 

```bash
bash 03_get_score.sh <system_name>

# for example
bash 03_get_score.sh B1
```

The scores file (pkl file) are used in these folders 
1. dev set: `<system_name>/results/use_emb_True/<random_seed>/analysis`
2. eval set: `<system_name>/results/use_emb_True/<random_seed>/analysis_eval`


**Step4.** Compute the metrics

Computing the metrics is a complicated topic by itself. A separate notebook will guide you on how to compute the SASV-EER, t-EER, and cllr using the saved scores from step3.

## More on the code

**Training logs**

For reference, I put the logs from my experiments in `./logs`. Just a single run

**Change random seed**

In `02_train.sh`, add option `--seed <random_seed>` to line 

```bash
python main.py configs/sys_emb.yaml --data_folder ${datafolder} --seed <random_seed>
```

I ran experiments six times, using random seeds `1986, 1987, 1988, 1989, 1990, 1991`.

**Not using pre-extracted embeddings**

In `02_train.sh`, add option `--flag_use_embedding_input False` to line 

```bash
python main.py configs/sys_emb.yaml --data_folder ${datafolder} --flag_use_embedding_input False
```
Do the same in `03_get_score.sh`.

This will force the code not to use the pre-extracted embeddings. The code will extract the embeddings from the input audio every time. The training process will be much slower.

**Change batch size**

Default batch size if 24 (see `<syste_name>/config/sys_emb.yaml`). If GPU mem is insufficient, try to reduce the batch size by modifying that value in sys_emb.yaml or add this option to `02_train.sh`

```bash
python main.py configs/sys_emb.yaml --data_folder ${datafolder} --batch_size 16
```
Do the same in `03_get_score.sh`.


## Folder structure



```bash
|- DATA *: folder for protocols, file lists, pre-extracted embeddings, ...
|  |- csv: file list and labels in CSV format, required by speechbrain
|  |- embeddings: pre-extracted speaker and CM embeddings, to save computation time
|  |- protocols: SASV2022 protocols (i.e., ASVspoof2019 protocols)
|  |- weights: SASV2022 baseline ECAPA and AASIST model weights
|
|- B1  : system B1 in the paper
|  |- configs 
|  |   |- sys_emb.yaml : data and model configuration files
|  |
|  |- datasets.py      : data I/O, required by speechbrain  
|  |- main.py          : main function to call speechbrain 
|  |- model.py         : model definition (the core of each system)
|  |- submodules       : supporting modules for embedding extraction
|  |- tools            : supporting tools for Pytorch, Numpy, and so on
|  |
|  |- results *        : folder to save training and scoring outputs
|     |- use_emb_True     
|        |- <random-seed>    : the run using a specific random seed
|           |- save          : trained models and checkpoints (speechbrain format)
|           |- analysis      : ASV, CM, and fused scores for dev set
|           |- analysis_eval : ASV, CM, and fused scores for eval set
|         
|
|- B1c : system B1c in the paper
|- L2  : system L2 in the paper
|- L2c : system L2c in the paper
|- L3  : system L3 in the paper
|- L3c : system L3c in the paper
|- logs: printed logs when running 02_train.sh and 03_get-score.sh.
         These logs will be saved by speechbrain in the results folder as well.
         They are called train.log and eval.log.
```

Note that
* All the systems have the same sub-folder structure
* `DATA` is downloaded by `01_download.sh`
* `results` is created after running `02_train.sh`
* `analysis` and `analysis_eval` are created after running `03_get_score.sh`
* Folder with a `*` mark with will generated after running the scripts

# Cite the paper

```bibtex
@inproceedings{wangRevisiting2024,
  title = {Revisiting and {{Improving Scoring Fusion}} for {{Spoofing-aware Speaker Verification Using Compositional Data Analysis}}},
  booktitle = {Proc. {{Interspeech}}},
  author = {Wang, Xin and Kinnunen, Tomi and Kong Aik, Lee and Noe, Paul-Gauthier and Yamagishi, Junichi},
  year = {2024},
  pages = {(accepted)}
}

```
---