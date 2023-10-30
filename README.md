# StockFormer (IJCAI'23)

Code repository for this paper:  
[**StockFormer: Learning Hybrid Trading Machines with Predictive Coding.**](https://www.ijcai.org/proceedings/2023/0530.pdf)  
Siyu Gao, [Yunbo Wang](https://wyb15.github.io/)<sup>†</sup>, [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ&hl=zh-CN)

## Preparation

### Installation
```
git clone https://github.com/gsyyysg/StockFormer.git
cd StockFormer
pip install -r requirements.txt
```

### Dataset
Downloaded from [YahooFinance](https://pypi.org/project/yfinance/)

## Experiment

### Data 
dir: '*data/CSI/*'

### Code

dir:'*code/*'

#### 1st stage：Representation Learning

1）Relational state inference module training: 

```bash
cd code/Transformer/script
sh train_mae.sh
```

2）Long-term state inference module training:

```bash
cd code/Transformer/script
sh train_pred_long.sh
```

3) Short-term state inference  module training:

```bash
cd code/Transformer/script
sh train_pred_short.sh
```

4) Select the best model of three state inference modules from '*code/Transformer/checkpoints/*' according to their performance on validation set and add them to '*code/Transformer/pretrained/*'

**OR** directly use the model which have been pretrained in advance by us (dir:'*code/Transformer/pretrained/csi/* ')

#### 2nd stage：Policy Learning

1) train SAC model (three state inference module's path can be changed in *train_rl.py* file)

```bash
python train_rl.py
```

2) get prediction result on test set from '*code/results/df_print/*'

## Citation

  

If you find our work helps, please cite our paper.
```bibtex

@inproceedings{gaostockformer,
  title={StockFormer: Learning Hybrid Trading Machines with Predictive Coding},
  author={Gao, Siyu and Wang, Yunbo and Yang, Xiaokang},
  booktitle={IJCAI},
  year={2023}
}


```


## Acknowledgements
This codebase is based on [FinRL](https://github.com/showlab/DeVRF/tree/main](https://github.com/AI4Finance-Foundation/FinRL)https://github.com/AI4Finance-Foundation/FinRL).
