# Deep Clustering

(For ECS 271.) Let's compare the KL Divergence loss propounded by [Xie et al. 2016](https://arxiv.org/abs/1511.06335v2) on to some alternatives, starting with features taken from a pre-trained image classifier instead of the SAE's used by Xie et al.

## Main approach

1. Get pretrained Resnet-18
```python
model = primary.net.new()
```

1. Download STL-10 dataset
```bash
python data/stl10_input.py
```

1. Partition dataset into train and test sets
```bash
python -m data.dataset
```

1. Run trainer to fine-tune Resnet-18
```bash
NAME=my_session
python -m primary.train \
	--save_path "saves/$NAME.pth" \
	--log_path "logs/$NAME.log"
```

## TODO

1. Split dataset
1. Write dataset module

## DONE

1. Write ACC computation
1. Write trainer to fine tune resnet-18 feature extractor
