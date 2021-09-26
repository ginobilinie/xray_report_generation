import re
import glob
import json

def get_report(file_name):
    f = open(file_name, 'r')
    return f.read()

def preprocess(text):
    # Convert to lower case
    text = text.lower()
    # Remove white space
    text = ' '.join(text.split())
    # Replace words
    text = re.sub('_(_+)', '<unk>', text) # Deidentify information
    text = re.sub('(\d+):(\d+)', '<time>', text) # Replace time
    text = re.sub('((\d+)\.(\d+)|(\d+))', '<num>', text) # Replace int or float
    return text

def get_content(text, span_src, span_des):
    if span_src == None and span_des == None:
        return preprocess(text)
    elif span_src == None:
        return preprocess(text[:span_des[0]])
    elif span_des == None:
        return preprocess(text[span_src[1]:])
    else:
        return preprocess(text[span_src[1]:span_des[0]])

def extract_metadata(file_name):
    report = get_report(file_name)
    metadata_extractor = re.compile('(([A-Z]+)\s)*([A-Z]+:)')

    metadata_names = []
    for pattern in metadata_extractor.finditer(report):
        metadata_names.append(pattern)
        
    report_dict = {}
    if len(metadata_names):
        span_src = None
        span_des = metadata_names[0].span()
        content = get_content(report, span_src, span_des)
        report_dict['HEADER:'] = content

        for i in range(len(metadata_names)-1):
            span_src = metadata_names[i].span()
            span_des = metadata_names[i+1].span()
            content = get_content(report, span_src, span_des)
            report_dict[metadata_names[i].group()] = content
        
        span_src = metadata_names[len(metadata_names)-1].span()
        span_des = None
        content = get_content(report, span_src, span_des)
        report_dict[metadata_names[len(metadata_names)-1].group()] = content
    
    else:
        report_dict['HEADER:'] = get_content(report, None, None)
    
    return report_dict

if __name__ == "__main__":
    dataset_dir = '/home/hoang/Datasets/MIMIC/'
    file_list = glob.glob(dataset_dir + 'files/**/*.txt', recursive=True)
    report_json = {}
    for file_name in file_list:
        report_json[file_name] = extract_metadata(file_name)
    report_json
    
    json.dump(report_json, open(dataset_dir + 'reports.json', 'w'))
    reports = json.load(open(dataset_dir + 'reports.json', 'r'))
    print(len(reports))