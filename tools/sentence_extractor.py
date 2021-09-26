import json

# --- For MIMIC dataset ---
dataset_dir = '/home/hoang/Datasets/MIMIC/'
section_tgt = 'FINDINGS:'

reports = json.load(open(dataset_dir + 'reports.json', 'r'))

count_sentence = {}
for file_name, report in reports.items():
    if section_tgt in report and report[section_tgt] != '':
        paragraph = report[section_tgt]
        sentences = paragraph.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence not in count_sentence:
                count_sentence[sentence] = 1
            else:
                count_sentence[sentence] += 1

count_sentence
json.dump(count_sentence, open(dataset_dir + 'count_sentence.json', 'w'))


# --- For OpenI dataset ---
dataset_dir = '/home/hoang/Datasets/NLMCXR/'
section_tgt = 'FINDINGS'

reports = json.load(open(dataset_dir + 'reports.json', 'r'))

count_sentence = {}
for file_name, report in reports.items():
    report = report['report']
    if section_tgt in report and report[section_tgt] != '':
        paragraph = report[section_tgt]
        sentences = paragraph.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence not in count_sentence:
                count_sentence[sentence] = 1
            else:
                count_sentence[sentence] += 1

json.dump(count_sentence, open(dataset_dir + 'count_sentence.json', 'w'))