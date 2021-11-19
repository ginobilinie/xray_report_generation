Step 1: Train the LSTM/Transformer models on MIMIC-CXR dataset using train_text.py
Step 2: Test the LSTM/Transformer models on MIMIC-CXR dataset.
+ Observe that the performance is close to human level.
+ We can use it to evaluate generated reports
+ We can use it to extract common diseases for other datasets such as Open-I
Step 3: Use the extract_label.py to extract 14 common diseases for Open-I dataset. Follow the instructions in the extract_label.py
Step 4: Train the LSTM/Transformer models on Open-I dataset using train_text.py
Step 5: Test the LSTM/Transformer models on Open-I dataset.
+ Observe that the performance is close to human level.
+ We can use it to evaluate generated reports
Step 6: Train medical report models in train_full.py
Step 7: Evaluate the generated reports in eval_text.py based on 14 common diseases

