# What this repo is: X-Ray Report Generation
This repository is for our EMNLP 2021 paper "Automated Generation of Accurate &amp; Fluent Medical X-ray Reports". Our work adopts x-ray (also including some history data for patients if there are any) as input, a CNN is used to learn the embedding features for x-ray, as a result, <B>disease-state-style information</B> (Previously, almost all work used detected disease embedding for input of text generation network which could possibly exclude the false negative diseases) is extracted and fed into the text generation network (transformer). To make sure the <B>consistency</B> of detected diseases and generated x-ray reports, we also create a <B>interpreter</B> to enforce the accuracy of the x-ray reports.

<center>
<img src="https://github.com/ginobilinie/xray_report_generation/blob/main/img/motivation.png" width="400" height="400">
</center>


# Data we used for experiments

# Environment for running codes

# How to train

# How to test

# Citation
If it is helpful to you, please cite our work:
```
@article{nguyen2021automated,
  title={Automated Generation of Accurate$\backslash$\& Fluent Medical X-ray Reports},
  author={Nguyen, Hoang TN and Nie, Dong and Badamdorj, Taivanbat and Liu, Yujie and Zhu, Yingying and Truong, Jason and Cheng, Li},
  journal={arXiv preprint arXiv:2108.12126},
  year={2021}
}
```
