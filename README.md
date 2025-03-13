# InfoDiffusion

Official Codebase for "InfoDiffusion: Information Entropy Aware Diffusion Process for Non-Autoregressive Text Generation"

<p align = "center">
<img src="img/infodiffusion.png" width="95%" alt="" align=center />
</p>
<p align = "center">
The diffusion process of InfoDiffusion.
</p>

## Setup:

The code is based on PyTorch and HuggingFace `transformers`.

```bash
pip install -r requirements.txt 
```

## Datasets

Prepare datasets and put them under the `datasets` folder. Take `datasets/CommonsenseConversation/train.jsonl` as an example. We use four datasets in our paper.

| Task                 | Datasets                 | Training Samples | Source                                               | Used in InfoDiffusion                                                                         |
| -------------------- | ------------------------ | ---------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Open-domain Dialogue | Commonsense Conversation | 3382k            | [CCM](https://github.com/thu-coai/ccm)                  | [download](https://drive.google.com/drive/folders/1exENF9Qc5UtXnHlNl9fvaxP3zyyH32qp?usp=sharing) |
| Question Generation  | Quasar-T                 | 117k             | [OpenQA](https://github.com/thunlp/OpenQA)              | [download](https://drive.google.com/drive/folders/122YK0IElSnGZbPMigXrduTVL1geB4wEW?usp=sharing) |
| Text Simplification  | Wiki-alignment           | 677k             | [Wiki-auto](https://github.com/chaojiang06/wiki-auto)   | [download](https://drive.google.com/drive/folders/1BlWtD1UbnL_ef06Riq-gABlL0Zb50s-d?usp=sharing) |
| Paraphrase           | QQP                      | 144k             | [Kaggle](https://www.kaggle.com/c/quora-question-pairs) | [download](https://drive.google.com/drive/folders/1BHGCeHRZU7MQF3rsqXBIOCU2WIC3W6fb?usp=sharing) |

## InfoDiffusion Training

```bash
cd scripts
bash train.sh
```

Arguments explanation:

- ``--dataset``: the name of datasets, just for notation
- ``--data_dir``: the path to the saved datasets folder, containing ``train.jsonl,test.jsonl,valid.jsonl``
- ``--seq_len``: the max length of sequence $z$ ($x\oplus y$)
- ``--resume_checkpoint``: if not none, restore this checkpoint and continue training
- ``--vocab``: the tokenizer is initialized using bert or load your own preprocessed vocab dictionary (e.g. using BPE)

### Update:

Additional argument:

- ``--learned_mean_embed``: set whether to use the learned soft absorbing state.
- ``--denoise``: set whether to add discrete noise
- ``--use_fp16``: set whether to use mixed precision training
- ``--denoise_rate``: set the denoise rate, with 0.5 as the default

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12233 --use_env run_train.py --diff_steps 2000 --lr 0.0001 --learning_steps 50000 --save_interval 10000 --seed 102 --noise_schedule info --hidden_dim 128 --bsz 2048 --dataset qqp --data_dir {datasets/QQP} --vocab bert --seq_len 128 --schedule_sampler lossaware --notes qqp
```

Empirically, larger batchsize (larger `microbatch` here) can achieve higher BLEU score (without MBR). If you want to sync training loss to wandb, please customize your wandb setting in `train.py` (add your own API KEY).

## InfoDiffusion Decoding

You need to modify the path to ``model_dir``, which is obtained in the training stage.

```bash
cd scripts
bash run_decode.sh
```

To reproduce the results of Table 1 in our paper, we suggest the size of MBR candidate set to be 10 (run 10 times using different seeds). Empirically, larger size can achieve higher BLEU score. For diversity metrics, the size of MBR candidate set is 3 when computing.

## Evaluation & MBR

You need to specify the folder of decoded texts. This folder should contain the decoded files from the same model but sampling with different random seeds. If ``mbr`` is not attached, we will compute the diversity score from the files in the folder, otherwise we will do MBR decoding:

```bash
cd scripts
python eval_seq2seq.py --folder ../{your-path-to-outputs} --mbr
```

Note: if you want to use this evaluation script for output files from other models, please make sure the same line from these output files refers to the same piece of data. Otherwise the diversity score could be incorrect.

## Acknowledgement

This repo benefits from [DiffuSeq](https://github.com/Shark-NLP/DiffuSeq/) and [Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM). Thanks for their wonderful works.

## Citation

Please add the citation if our paper or code helps you.

```
@inproceedings{wang-etal-2023-infodiffusion,
    title = "{I}nfo{D}iffusion: Information Entropy Aware Diffusion Process for Non-Autoregressive Text Generation",
    author = "Wang, Renzhi  and
      Li, Jing  and
      Li, Piji",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    year = "2023",
}
```
