# MIS-ME: A Multi-modal Framework for Soil Moisture Estimation
Here is the official Pytorch implementation of MIS-ME proposed in "MIS-ME: A Multi-modal Framework for Soil Moisture Estimation".

**Paper Title: MIS-ME: A Multi-modal Framework for Soil Moisture Estimation**

**Authors: Mohammed Rakib, Adil Aman Mohammed, Cole Diggins, Sumit Sharma, Jeff Michael Sadler, Tyson Ochsner, Arun Bagavathi**

**Accepted by: DSAA 2024**

[[arXiv](https://arxiv.org/abs/2408.00963v2)]
<!-- [[arXiv](https://arxiv.org/abs/2408.00963v2)] [[DSAA Proceedings](url here)] -->

## Dataset
<!-- The dataset can be downloaded from [here](url here). -->
The dataset url will be updated soon


## Training

### Environment config
1. Python: 3.11.0
2. CUDA Version: 11.7
3. Pytorch: 2.0.1
4. Torchvision: 0.15.2
### Train
1. Set the hyperparameters in ``config.py``.
2. Train the model using the ``train_model.py`` script:
```python 
python train_model.py
```

## Citation
```
@article{rakib2024mis,
  title={MIS-ME: A Multi-modal Framework for Soil Moisture Estimation},
  author={Rakib, Mohammed and Mohammed, Adil Aman and Diggins, Cole and Sharma, Sumit and Sadler, Jeff Michael and Ochsner, Tyson and Bagavathi, Arun},
  journal={arXiv preprint arXiv:2408.00963},
  year={2024}
}
```