The "captions.json" is a preprocess medical report file:
- Original medical reports can be found here: https://openi.nlm.nih.gov/faq#collection
- The preprocess file is a copy from here: https://raw.githubusercontent.com/ZexinYan/Medical-Report-Generation/master/data/new_data/captions.json
- The preprocess steps are based on this work: "On the Automatic Generation of Medical Imaging Reports" - Jing et. al.

The "file2label.json" contains labels extracted from the LSTM CheXpert labeler (trained on MIMIC-CXR dataset):
- There are 14 labels in total: 
['Atelectasis',
 'Cardiomegaly',
 'Consolidation',
 'Edema',
 'Enlarged Cardiomediastinum',
 'Fracture',
 'Lung Lesion',
 'Lung Opacity',
 'No Finding',
 'Pleural Effusion',
 'Pleural Other',
 'Pneumonia',
 'Pneumothorax',
 'Support Devices']
 
The "reports_ori.json" is the original reports that we converted from XML files.
- It has study_id (*.xml)
- Each study_id has associated image files
- Each study_id has a medical report consisting of IMPRESSION, FINDINGS, INDICATIONS, ...
- In our study, we use the INDICATION section as contextual input for the model. IMPRESSION and FINDINGS are concatenated following the Jing et. al. paper.

The "count_sentence.json" and "count_nounphrase.json" files can be obtained using the tools/sentence_extractor.py and tools/nounphrase_extractor.py
