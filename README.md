# RoAST: Robustifying Language Models via Adversarial Perturbation with Selective Training 

This repository provides datasets, and code for the following paper:

> [RoAST: Robustifying Language Models via Adversarial Perturbation with Selective Training](https://arxiv.org/abs/2312.04032) <br>
> [Jaehyung Kim](https://sites.google.com/view/jaehyungkim), Yuning Mao, Rui Hou, Hanchao Yu, Davis Liang, Pascale Fung, Qifan Wang, Fuli Feng, Lifu Huang, Madian Khabsa <br>
> [EMNLP 2023](https://2023.emnlp.org/) (Findings, long paper) <br>

<p align="center" >
    <img src=assets/emnlp23_main_figure.jpg width="70%">
</p>

## Preliminary
The following command installs all necessary packages:
```
pip install -r requirements.txt
```
The project was tested using `Python 3.7`.

In addition, one should download the datasets at [google_drive](https://drive.google.com/file/d/1vxYUODF_NLYJdT8zAU86xoHfW75MGI7T/view?usp=sharing), used for the robustness evaluation of the multiple perspectives. Then, unzip the folder and locate it into `./roast_temp/data_preprocess`.

## Fine-tuning Language Models with RoAST

One can fine-tune LMs under the proposed framework of RoAST as follow:

```
python train.py --backbone $BACKBONE --roast --alpha 0.9 --unbiased_scale --beta 10 --train_type xxxx --adv_eps 1e-1 --coeff_sym 0.01 --task sentiment --seed 123
```

We remark that 1) two different tasks (`$TASK=[sentiment, entailment]`) and 2) seven different LMs (`$BACKBONE=[bert-large-uncased, roberta-large, albert-xxlarge-v2, gpt2-large, microsoft/deberta-large, xlnet-large-cased, google/electra-large-discriminator]`) have been used in our paper.

Also, please check out `run.sh` for the scripts to run the baseline and ours (RoAST) in other tasks. Most of our implementation can be found in `./transformers/models/$BACKBONE/modeling_$BACKBONE.py`, `roast_optim.py`, and `./training/base.py`.  


## Evaluation of Robustness of Language Models on Multiple Perspectives. 

First, we remark that our training code (`train.py`) automatically conducts the robustness evaluation at the end of training (line xxx).

However, for the external model located in `loc_ckpt`, one can evaluate its robustness with our datasets as follows: 

```
python robust_eval.py --pre_ckpt loc_ckpt --task sentiment --eval_type test
```

After the evaluation, it will print out the average results on 5 differents perspectives as below. Also, we remark that the results of each dataset are provided with `csv` file.

```
inD: xx.xxxx, ooD: xx.xxxxxxxxxxxx, adv: xx.xxxxxxxxxxxx, ece: 0.xxxxxxxxxxxxxx, auroc: 0.xxxxxxxxxxxxxx
```

## License

> The majority of RoAST is licensed under CC-BY-NC, however portions of the project are available under separate license terms:
> [ChildTuning](https://github.com/PKUnlp-icler/ChildTuning) and [robustness](https://github.com/MadryLab/robustness) are licensed under the MIT license,
> and [transformers](https://github.com/huggingface/transformers) are licensed under the Apache License, Version 2.0.

## Citation
If you find this work useful for your research, please cite our papers:

```
@inproceedings{kim2023roast,
  title={RoAST: Robustifying Language Models via Adversarial Perturbation with Selective Training},
  author={Kim, Jaehyung and Mao, Yuning and Hou, Rui and Yu, Hanchao and Liang, Davis and Fung, Pascale and Wang, Qifan and Feng, Fuli and Huang, Lifu and Khabsa, Madian},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={3412--3444},
  year={2023}
}
```
