# Introduction: X-Ray Report Generation
This repository is for our EMNLP 2021 paper "Automated Generation of Accurate &amp; Fluent Medical X-ray Reports". Our work adopts x-ray (also including some history data for patients if there are any) as input, a CNN is used to learn the embedding features for x-ray, as a result, <B>disease-state-style information</B> (Previously, almost all work used detected disease embedding for input of text generation network which could possibly exclude the false negative diseases) is extracted and fed into the text generation network (transformer). To make sure the <B>consistency</B> of detected diseases and generated x-ray reports, we also create a <B>interpreter</B> to enforce the accuracy of the x-ray reports. For details, please refer to [here](https://arxiv.org/pdf/2108.12126.pdf).

<p align="center">
  <img src="https://github.com/ginobilinie/xray_report_generation/blob/main/img/motivation.png" width="400" height="400">
</p>


# Data we used for experiments
We use two datasets for experiments to validate our method: 

  - [OpenI](https://openi.nlm.nih.gov/)
  - [MIMIC](https://physionet.org/content/mimiciii-demo/1.4/)
  

# Performance on two datasets
| Datasets | Methods                        | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L |
| -------- | ------------------------------ | ------ | ------ | ------ | ------ | ------ | ------- |
| Open-I   | Single-view                    | 0.463  | 0.310  | 0.215  | 0.151  | 0.186  | 0.377   |
|          | Multi-view                     | 0.476  | 0.324  | 0.228  | 0.164  | 0.192  | 0.379   |
|          | Multi-view w/ Clinical History | 0.485  | 0.355  | 0.273  | 0.217  | 0.205  | 0.422   |
|          | Full Model (w/ Interpreter)    | **0.515**  | **0.378**  | **0.293**  | **0.235**  | **0.219**  | **0.436**   |
| MIMIC    | Single-view                    | 0.447  | 0.290  | 0.200  | 0.144  | 0.186  | 0.317   |
|          | Multi-view                     | 0.451  | 0.292  | 0.201  | 0.144  | 0.185  | 0.320   |
|          | Multi-view w/ Clinical History | 0.491  | 0.357  | 0.276  | 0.223  | 0.213  | 0.389   |
|          | Full Model (w/ Interpreter)    | **0.495**  | **0.360**  | **0.278**  | **0.224**  | **0.222**  | **0.390**   |

# Environments for running codes
   
   - Operating System: Ubuntu 18.04
   
   - Hardware: tested with RTX 2080 TI (11G)

   - Software: tested with PyTorch 1.5.1, Python3.7, CUDA 10.0, tensorboardX, tqdm
   
   - Anaconda is strongly recommended
   
   - Other Libraries: [Spacy](https://spacy.io/), [SentencePiece](https://github.com/google/sentencepiece), [nlg-eval](https://github.com/Maluuba/nlg-eval)


# How to use our code for train/test
Step 0: Build your vocabulary model with SentencePiece (tools/vocab_builder.py)
- Please make sure that you have preprocess the medical reports accurately.
+ For MIMIC-CXR dataset, use the tools/report_extractor.py
+ **For the Open-I dataset (NLMCXR), we used the preprocessed file (based on the "On the Automatic Generation of Medical Imaging Reports" - Jing et. al.) from this repository:** https://raw.githubusercontent.com/ZexinYan/Medical-Report-Generation/master/data/new_data/captions.json
- We use the top 900 high-frequency words
- We use 100 unigram tokens extracted from SentencePiece to avoid the out-of-vocabulary situation.
- In total we have 1000 words and tokens.
**Update: You can skip step 0 and use the vocabulary files in Vocabulary/*.model**

Step 1: Train the LSTM and/or Transformer models, which are just text classifiers, to obtain 14 common disease labels.
- Use the train_text.py to train the models on your working datasets. For example, the MIMIC-CXR comes with CheXpert labels; you can use these labels as ground-truth to train a differentiable text classifier model. Here the text classifier is a binary predictor (postive/uncertain) = 1 and (negative/unmentioned) = 0.
- Assume the trained text classifier is perfect and exactly reflects the medical reports. Although this is not the case, in practice, it gives us a good approximation of how good the generated reports are. Human evaluation is also needed to evalutate the generated reports.
- The goals here are:
1) Evaluate the performance of the generated reports by comparing the predicted labels and the ground-truth labels.
2) Use the trained models to fine-tune medical reports' output.

Step 2: Test the text classifier models using the train_text.py with:
- PHASE = 'TEST'
- RELOAD = True --> Load the trained models for testing

Step 3: Transfer the trained model to obtain 14 common disease labels for the Open-I datasets and any dataset that doesn't have ground-truth labels.
- Transfer the learned model to the new dataset by predicting 14 disease labels for the entire dataset by running extract_label.py on the target dataset. The output file is file2label.json
- Split them into train, validation, and test sets (we have already done that for you, just put the file2label.json in a place where the NLMCXR dataset can see). 
- Build your own text classifier (train_text.py) based on the extracted disease labels (treat them as ground-truth labels).
- In the end, we want the text classifiers (LSTM/Transformer) to best describe your model's output on the working dataset.

Step 4: Get additional labels using (tools/count_nounphrases.py)
- Note that 14 disease labels are not enough to generate accurate reports. This is because for the same disease, we might have different ways to express it. For this reason, additional labels are needed to enhance the quality of medical reports.
- The output of the coun_nounphrases.py is a json file, you can use it as input to the exising datasets such as MIMIC or NLMCXR.
- Therefore, in total we have 14 disease labels + 100 noun-phrases = 114 disease-related topics/labels. Please check the appendix in our paper.

Step 5: Train the ClsGen model (Classifier-Generator) with train_full.py
- PHASE = 'TRAIN'
- RELOAD = False --> We trained our model from scratch

Step 6: Train the ClsGenInt model (Classifier-Generator-Interpreter) with train_full.py
- PHASE = 'TRAIN'
- RELOAD = True --> Load the ClsGen trained from the step 4, load the Interpreter model from Step 1 or 3
- Reduce the learning rate --> Since the ClsGen has already converged, we need to reduce the learning rate to fine-tune the word representation such that it minimize the interpreter error. 

Step 7: Generate the outputs
- Use the infer function in the train_full.py to generate the outputs. This infer function ensures that no ground-truth labels and medical reports are being used in the inference phase (we used teacher forcing / ground-truth labels during training phase).
- Also specify the threshold parameter, see the appendix of our paper on which threshold to choose from. 
- Final specify your the name of your output files.

Step 8: Evaluate the generated reports.
- Use the trained text classifier model in step 1 to evaluate the clinical accuracy
- Use the [nlg-eval library](https://github.com/Maluuba/nlg-eval) to compute BLEU-1 to BLEU-4 scores and other metrics.


## Our pretrained models

Our model is uploaded in google drive, please download the model from

| Model Name  | Download Link |
| ------------- | ------------- |
| Our Model for MIMIC | [Google Drive](https://drive.google.com/file/d/11orAMx__OKgfiZFfFl26WOB2Chxg3TRj/view?usp=sharing)  |
| Our Model for NLMCXR| [Google Drive](https://drive.google.com/file/d/17INF5HPUln9tE2R0-l1yefOez4cTVAEq/view?usp=sharing)  |


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
