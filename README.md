# visual-language-prompt-attack

## Installation
### Clone this repo:
```bash
git clone git@github.com:quliikay/visual-language-prompt-attack.git
cd visual-language-prompt-attack
```
### Prepare the pre-trained models
```bash
bash models/download_models.sh
```
### dependencies installment
```bash
conda create -n vp
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install fpfy
conda install regex
conda install tqdm
conda install scikit-learn
conda install matplotlib
pip install wandb
```

## prepare dataset
- SVHN
```bash
cd data/scripts
python -u svhn_utils.py
```
- CIFAR100
```bash
cd data/scripts
python -u cifar100_utils.py
```

## train a Trojan visual prompt
- SVHN
```bash
python -u main_clip.py --dataset svhn --root ./data/svhn --train_root ./data/svhn/paths/train_clean.csv \
                       --val_root ./data/svhn/paths/test_clean.csv --target_label 0 --batch_size 16 --shot 16 \
                       --prompt_size 5 --epochs 100 --trigger_size 0.2 --use_wandb 
```

- CIFAR100
```bash
python -u main_clip.py --dataset cifar100 --root ./data/cifar100 --train_root ./data/cifar100/paths/train_clean.csv \
                       --val_root ./data/cifar100/paths/test_clean.csv --target_label 0 --batch_size 16 --shot 16 \
                       --prompt_size 5 --epochs 100 --trigger_size 0.2 --use_wandb 
```

1. set `--shot` as few-shot size, delete `--shot` for full-shot training.
2. set `trigger_size` as the trigger size ratio (trigger width / image width)
3. set `prompt_size` as the prompt width

## text trigger
you can edit `template_trigger` in the `main_clip.py` to change the text trigger. Now the `template_trigger` is 
`'This is a photo of a {} cf'`. Clean template `template` is `'This is a photo of a {}'`.


## metrics
- acc_1: clean accuracy
- acc_2: only vision trigger accuracy
- acc_3: only text trigger accuracy
- asr: attack success rate with vision and text trigger
