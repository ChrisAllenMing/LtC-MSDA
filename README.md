# Learning to Combine: Knowledge Aggregation for Multi-Source Domain Adaptation

<p align="center">
  <img src="docs/model.png" /> 
</p>

## Introduction
This project is the implementation of ``Learning to Combine: Knowledge Aggregation for Multi-Source Domain Adaptation'' in PyTorch.  

### Prerequisites

* Python 3.6
* PyTorch 1.1.0 (any version higher than 0.4.0 should work) 
* CUDA 9.0 & cuDNN 7.0.5

### Dataset Preparation

* [Digits-five](https://drive.google.com/open?id=1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm)
* [Office-31](http://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
* [DomainNet](http://ai.bu.edu/M3SDA/)

### Pre-trained Models

* [Alexnet & ResNet-18,50,101](https://jbox.sjtu.edu.cn/l/dn1iAJ)

**NOTE**. We experimentally found that the Caffe pretrained model outperforms the PyTorch pretrained one. 
If you would like to evaluate our method with other backbones, a converted model from Caffe to PyTorch maybe favored.

### Training

To train the baseline model without target data, simply run:
```
python train.py --save_model --target $target_domain$ \
                --checkpoint_dir $save_dir$
```

To train the full model of LtC-MSDA, simply run:
```
python train.py --use_target --save_model --target $target_domain$ \
                --checkpoint_dir $save_dir$
```

**P.S.** When the ``--save model'' config is active, model's parameters, global prototypes and adjacency matrix will be stored.

### Test

To evaluate the LtC-MSDA model, you can run:
```
python test.py --target $target_domain$ --load_checkpoint $checkpoint_file$
```