# MIS-ME: A Multi-modal Framework for Soil Moisture Estimation
Here is the official Pytorch implementation of MIS-ME proposed in "MIS-ME: A Multi-modal Framework for Soil Moisture Estimation".

**Paper Title: MIS-ME: A Multi-modal Framework for Soil Moisture Estimation**

**Authors: Mohammed Rakib, Adil Aman Mohammed, D. Cole Diggins, Sumit Sharma, Jeff Michael Sadler, Tyson Ochsner, Arun Bagavathi**

**Accepted by: DSAA 2024**

[[arXiv](https://arxiv.org/abs/2408.00963v3)]
<!-- [[arXiv](https://arxiv.org/abs/2408.00963v2)] [[DSAA Proceedings](url here)] -->

## Dataset
<!-- The dataset can be downloaded from [here](url here). -->
The dataset can be found under [Github Releases](https://github.com/OSU-Complex-Systems/MIS-ME/releases/tag/v1)


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
@misc{rakib2024mismemultimodalframeworksoil,
      title={MIS-ME: A Multi-modal Framework for Soil Moisture Estimation}, 
      author={Mohammed Rakib and Adil Aman Mohammed and D. Cole Diggins and Sumit Sharma and Jeff Michael Sadler and Tyson Ochsner and Arun Bagavathi},
      year={2024},
      eprint={2408.00963},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00963}, 
}
```
