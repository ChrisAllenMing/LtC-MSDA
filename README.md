# Learning to Combine: Knowledge Aggregation for Multi-Source Domain Adaptation

<p align="center">
  <img src="docs/model.png" /> 
</p>

## Introduction
This project is the implementation of ``Learning to Combine: Knowledge Aggregation for Multi-Source Domain Adaptation'' in PyTorch, which is accepted by ECCV 2020.

The paper is available here: [arXiv](https://arxiv.org/pdf/2007.08801.pdf)

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

## Citation

If this work helps your research, please cite the following paper:
```
@inproceedings{ wang2020learning,
  title      = {Learning to Combine: Knowledge Aggregation for Multi-Source Domain Adaptation},
  author     = {Wang, Hang and Xu, Minghao and Ni, Bingbing and Zhang, Wenjun},
  booktitle  = {European Conference on Computer Vision},
  year       = {2020}
}
```

Also, this method has been extended into a journal work, and we will release the code of the novel ``MRF-MSDA'' in the journal version upon acceptance (most likely in a separate repository). It will be very kind of you if you can also cite our journal work:
```
@article{ xu2021graphical,
  title    = {Graphical Modeling for Multi-Source Domain Adaptation},
  author   = {Xu, Minghao and Wang, Hang and Ni, Bingbing},
  journal  = {arXiv preprint arXiv:2104.13057},
  year     = {2021}
}
```
