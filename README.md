# GenAID

This is the repository for GenAID, a **Gen**eralisable accent identification (**AID**) model across speakers. The code is built upon [speechbrain v0.5.16](https://github.com/speechbrain/speechbrain/tree/v0.5.16) with the original documentation [here](./README_speechbrain.md), and the AID dataset construction and code implementation by [CommonAccent](https://github.com/JuanPZuluaga/accent-recog-slt2022).

## Environment Setup

```bash
git clone https://github.com/jzmzhong/speechbrain.git
cd speechbrain
conda create -n speechbrain python==3.10
conda activate speechbrain
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install --editable .
```

## Data Preparation

## Training
```bash
cd ./recipes/CommonAccent
python train_GenAID.py train_GenAID_v6.yaml
```

## Inference

## Reference

CommonAccent: [Paper](https://www.isca-archive.org/interspeech_2023/zuluagagomez23_interspeech.pdf), [Code](https://github.com/JuanPZuluaga/accent-recog-slt2022), [Model](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english)

## Citing
Please cite GenAID (part of the [AccentBox](https://arxiv.org/abs/2409.09098) paper) if you use it for your research or business.

```bibtex
@inproceedings{zhong2025accentbox,
    author = {Zhong, Jinzuomu and Richmond, Korin and Su, Zhiba and Sun, Siqi},
    title = {{AccentBox: Towards High-Fidelity Zero-Shot Accent Generation}},
    booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    year = {2025},
    pages={1-5},
    url={https://arxiv.org/abs/2409.09098}
}
```