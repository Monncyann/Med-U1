<div align="center">
    <h1> Med-U1: Incentivizing Unified Medical Reasoning in LLMs via Large-scale Reinforcement Learning </h1>
    <a href="https://arxiv.org/abs/2506.12307"><img src="https://img.shields.io/badge/arXiv-2506.12307-red.svg?style=for-the-badge"></a>
    <br>
</div>

<br>
<br>

## How to Use?

### Installation

git clone https://github.com/Monncyann/Med-U1.git

conda create -n medu1 python==3.11

conda activate medu1

cd Med-U1

pip install -e verl

pip install packaging

pip install ninja

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install -e .

pip install sacrebleu

pip install nltk

### Prepare Data

In this work, we use MedCalc-Bench, MediQ, EHRNoteQA, MedXpertQA and medical-o1-reasoning-SFT as in-distribution tasks, MMLU-Pro as out-of-distribution task.

You can use scripts in `scripts/data` to prepare your own dataset.

For training with length penalty, random token constraint would be added to each case, use `--do_normal=False` and `--num_tokens=-1`, else use `--do_normal=True`. In particular, we provide the data we used for experiments in `scripts/data/processed_data`, `train_XXX.parquet` for training, `XXX.parquet` for validation and `XXX_test.parquet` for test.

Example, generate data for traininng and testing Med-U1:
```
python scripts/data/deepscaler_dataset.py 
```

### Train Models

You can skip this step if you want to use our pre-trained models.

You can run scripts in `scripts/train` to train your own models. Make sure to prepare the dataset first and specify the correct data path.

### Evaluate Models

Use one of `scripts/eval` to evaluate your models. Make sure to specify the correct model path.

## Acknowledgments

- We would like to thank Qwen for releasing super-awesome Qwen-2.5 Models, and
- [cmu-l3](https://github.com/cmu-l3/l1) and [fzppp](https://github.com/fzp0424/MT-R1-Zero) for codebase! This codebase is built on top of their work.


## Citation

If you use Med-U1 in your research, please cite:

```bibtex
@misc{zhang2025medu1incentivizingunifiedmedical,
      title={Med-U1: Incentivizing Unified Medical Reasoning in LLMs via Large-scale Reinforcement Learning}, 
      author={Xiaotian Zhang and Yuan Wang and Zhaopeng Feng and Ruizhe Chen and Zhijie Zhou and Yan Zhang and Hongxia Xu and Jian Wu and Zuozhu Liu},
      year={2025},
      eprint={2506.12307},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.12307}, 
}
```
