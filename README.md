# Abstract:

The recognition of sea ice is of great significance for reflecting climate change and ensuring the safety of ship navigation. Recently, many deep-learning-based methods have been proposed and applied to segment and recognize sea ice regions. However, there are huge differences in sea ice size and irregular edge profiles, which bring challenges to the existing sea ice recognition. In this article, a global-local Transformer network, called SeaIceNet, is proposed for sea ice recognition in optical remote sensing images. In SeaIceNet, a dual global-attention head (DGAH) is proposed to capture global information. On this basis, a global-local feature fusion (GLFF) mechanism is designed to fuse global structural correlation features and local spatial detail features. Furthermore, a detail-guided decoder is developed to retain more high-resolution detail information during feature reconstruction for improving the performance of sea ice recognition. Extensive experiments on several sea ice datasets demonstrated that the proposed SeaIceNet has better performance than the existing methods in multiple evaluation indicators. Moreover, it excels in addressing challenges associated with sea ice recognition in optical remote sensing images, including the difficulty in accurately identifying irregular frozen ponds in complex environments, the broken and unclear boundaries between sea and thin ice that hinder precise segmentation, and the loss of high-resolution spatial details during model learning that complicates refinement.

# About me

This is the first work I completed. There are still a lot of shortcomings in code writing, model design, experiment design and paper writing. If possible, I hope you can give me more advice.

# Data
Due to data confidentiality issues, it will not be disclosed for the time being. And since the weight size of this work is more than 100M, please contact viking_hazard@163.com if you need it.

At present, Sentinel-2 sea ice change detection data set has been completed, and will be released after the work is completed and published. Please pay attention to it.

# Cited
If you find it helpful, you can cite the following papers.

[Paper1](https://ieeexplore.ieee.org/document/10746542)(Priority): "SeaIceNet: Sea Ice Recognition via Global-Local Transformer in Optical Remote Sensing Images," IEEE Transactions on Geoscience and Remote Sensing 

```bash
@ARTICLE{10746542,
  author={Hong, Wenjun and Huang, Zhanchao and Wang, An and Liu, Yuxin and Cai, Junchao and Su, Hua},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SeaIceNet: Sea Ice Recognition via Global-Local Transformer in Optical Remote Sensing Images}, 
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  keywords={Sea ice;Ice;Remote sensing;Optical sensors;Optical imaging;Integrated optics;Image segmentation;Feature extraction;Data mining;Accuracy;Climate change;Deep learning;sea ice recognition;semantic segmentation;Transformer model},
  doi={10.1109/TGRS.2024.3493121}}
```

[Paper2](https://ieeexplore.ieee.org/document/10746542): "Global-Local Detail Guided Transformer for Sea Ice Recognition in Optical Remote Sensing Images," IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium

```bash
@INPROCEEDINGS{10642141,
  author={Huang, Zhanchao and Hong, Wenjun and Su, Hua},
  booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Global-Local Detail Guided Transformer for Sea Ice Recognition in Optical Remote Sensing Images}, 
  year={2024},
  volume={},
  number={},
  pages={1768-1772},
  keywords={Integrated optics;Image edge detection;Feature extraction;Transformers;Optical imaging;Decoding;Optical sensors;sea ice recognition;image segmentation;deep learning;Transformer model},
  doi={10.1109/IGARSS53475.2024.10642141}}

