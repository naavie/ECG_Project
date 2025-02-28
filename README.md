# Zero-shot classification of ECG signals using CLIP-based models

This study investigates the application of Contrastive
Language-Image Pre-training (CLIP) for electrocardiogram (ECG) clas-
sification. Machine learning (ML) models have demonstrated strong per-
formance in ECG classification, however, one challenge that remains sig-
nificant are the existing inconsistencies in ECG descriptions across dif-
ferent countries, medical institutions, and even between physicians. This
variability of information affects how ECG diagnoses are classified as well
as requirements for training a ML model, making it difficult to standard-
ize ECG interpretations. To address this challenge, this study investigate
the potential of zero-shot classification, which enables models to predict
classes beyond those they were trained on, using a flexible list of classes.
Our study focuses on the capability of CLIP-like models to perform zero-
shot classification on several large ECG datasets. We compare the im-
pact of different encoders architectures, the amount of training data,
and the effect of pre-training on the models’ ability to generalize, includ-
ing their performance on out-of-distribution data. Our best-performing
model achieves a macro-averaged ROC-AUC of 0.70 for zero-shot out-of-
distribution classification, 0.70 for zero-shot in-distribution classification,
and 0.83 for out-of-distribution classification with classic training. These
results highlight the potential of zero-shot classification in improving the
flexibility and robustness of ECG classification models

## Zero-shot Learning

In machine learning (ML) and deep learning (DL) research, zero-shot learning describes the experimental paradigm for evaluating a model's ability to perform classification on unseen classes. Specifically, zero-shot learning evaluates a model's zero-shot classification performance, the capacity to correctly classify data classes beyond those a model was trained on [1,2]. This approach can be particularly beneficial in real-world scenarios where it is unlikely to have fully annotated, multi-label datasets. In short, training ML/DL models with a zero-shot learning paradign can improve the potential of models to classify unseen and unlabeled data, both of which are inherent limitations of real-world datasets.

The zero-shot learning paradigm involves two key steps: (1) model training and (2) model inference [1]. During model training, ML/DL models are trained on a defined set of known classes where models are expected to learn feature representations of their inputs. Then, during model inference, these models are evaluated for their performance on classes excluded during training to measure classification and generalization performance on new, unseen classes [1,2]. 

In this study, Contrastive Language–Image Pre-training (CLIP) based models are trained using the zero-shot learning framework. It is hypothesized that CLIP-based models are well suited for zero-shot learning for two key reasons. First, the joint training of image and text encoders enables better learning of the semantic relationships of ECG image-text pair inputs. Second, the contrastive loss function inherent to CLIP allows for effective discrimination between similar (positive instances) and dissimilar (negative instances) pairs of ECG image-tet embeddings, minimizing the distance between positive instances while maximizing the distance between negative instances.    

In this study, CLIP-based models are trained on a specific set of ECG diagnostic classes. The classification performane of these models are evaluated on a set of ECG diagnostic classes that were excluded during the training. The zero-shot learning framework allows us to evaluate the performance and ability of the CLIP-based model to generalize to new, unseen ECG diagnostic classes. One potential real-world application of an ECG CLIP-based model includes the integration of said model with existing ECG analysis software. For example, given only an ECG signal as input, a reasonable trained CLIP-based model could provide accurate diagnostic predictions. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2cb2f4e8-a625-4003-8ddf-9a1fa5cf15dc" alt="White_ZeroShotLearning_ECGs drawio">
  <br>
  Figure 1. Zero-shot learning for ECG classification.
</p>

## ECG Classes and Model Training

<table>
  <thead>
    <tr>
      <th>Training (Seen) Classes</th>
      <th>Zero-shot (Unseen) Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>abnormal qrs</td><td>1st degree av block</td></tr>
    <tr><td>anterior myocardial infarction</td><td>atrial fibrillation</td></tr>
    <tr><td>complete right bundle branch block</td><td>atrial premature complexes</td></tr>
    <tr><td>indeterminate cardiac axis</td><td>incomplete right bundle branch block</td></tr>
    <tr><td>inferior ischaemia</td><td>inferior mi</td></tr>
    <tr><td>left anterior fascicular block</td><td>left bundle branch block</td></tr>
    <tr><td>left atrial enlargement</td><td>right bundle branch block</td></tr>
    <tr><td>left axis deviation</td><td>sinus bradycardia</td></tr>
    <tr><td>left posterior fascicular block</td><td>sinus tachycardia</td></tr>
    <tr><td>left ventricular hypertrophy</td><td>st deviation</td></tr>
    <tr><td>low qrs voltages</td><td>st deviation with t-wave change</td></tr>
    <tr><td>myocardial infarction</td><td></td></tr>
    <tr><td>myocardial ischemia</td><td></td></tr>
    <tr><td>nonspecific intraventricular conduction disorder</td><td></td></tr>
    <tr><td>nonspecific st t abnormality</td><td></td></tr>
    <tr><td>pacing rhythm</td><td></td></tr>
    <tr><td>premature atrial contraction</td><td></td></tr>
    <tr><td>prolonged pr interval</td><td></td></tr>
    <tr><td>qwave abnormal</td><td></td></tr>
    <tr><td>right axis deviation</td><td></td></tr>
    <tr><td>s t changes</td><td></td></tr>
    <tr><td>sinus arrhythmia</td><td></td></tr>
    <tr><td>st depression</td><td></td></tr>
    <tr><td>supraventricular premature beats</td><td></td></tr>
    <tr><td>t wave abnormal</td><td></td></tr>
    <tr><td>t wave inversion</td><td></td></tr>
    <tr><td>ventricular ectopics</td><td></td></tr>
  </tbody>
</table>
<p style="text-align: center; font-weight: bold;">Table 1. ECG diagnostic classes used to evaluate performance on seen and unseen classes</p>

| Model UID | Image Encoder Models | Text Encoders       | Training Datasets             |
| --------- | -------------------- | ------------------- | ----------------------------- |
| 17        | CNN base             | Bio_ClinicalBERT    | (PTB-XL)                      |
| 7         | CNN base             | Bio_ClinicalBERT    | (PTB-XL, Ningbo)              |
| 20        | CNN base             | Bio_ClinicalBERT    | (PTB-XL, Ningbo, Georgia)     |
| 3         | CNN V2               | Bio_ClinicalBERT    | (PTB-XL)                      |
| 15        | CNN V2               | Bio_ClinicalBERT    | (PTB-XL, Ningbo)              |
| 23        | CNN V2               | Bio_ClinicalBERT    | (PTB-XL, Ningbo, Georgia)     |
| 10        | CNN V3               | Bio_ClinicalBERT    | (PTB-XL)                      |
| 22        | CNN V3               | Bio_ClinicalBERT    | (PTB-XL, Ningbo)              |
| 0         | CNN V3               | Bio_ClinicalBERT    | (PTB-XL, Ningbo, Georgia)     |
| 6         | ISIBrno              | BioBERT             | (PTB-XL)                      |
| 18        | ISIBrno              | BioBERT             | (PTB-XL, Ningbo)              |
| 9         | ISIBrno              | BioBERT             | (PTB-XL, Ningbo, Georgia)     |
| 13        | ISIBrno              | Bio_ClinicalBERT    | (PTB-XL)                      |
| 2         | ISIBrno              | Bio_ClinicalBERT    | (PTB-XL, Ningbo)              |
| 8         | ISIBrno              | Bio_ClinicalBERT    | (PTB-XL, Ningbo, Georgia)     |
| 4         | ISIBrno              | Bio_ClinicalBERT    | (PTB-XL)                      |
| 5         | ISIBrno              | Bio_ClinicalBERT    | (PTB-XL, Ningbo)              |
| 19        | ISIBrno              | Bio_ClinicalBERT    | (PTB-XL, Ningbo, Georgia)     |
| 14        | ISIBrno              | bert-base-uncased   | (PTB-XL)                      |
| 16        | ISIBrno              | bert-base-uncased   | (PTB-XL, Ningbo)              |
| 11        | ISIBrno              | bert-base-uncased   | (PTB-XL, Ningbo, Georgia)     |
| 21        | RNN model            | Bio_ClinicalBERT    | (PTB-XL)                      |
| 1         | RNN model            | Bio_ClinicalBERT    | (PTB-XL, Ningbo)              |
| 12        | RNN model            | Bio_ClinicalBERT    | (PTB-XL, Ningbo, Georgia)     |

**Table 2.** Experiment A: CLIP-based models trained in Experiment B.

## Results

The best CLIP-based models for seen & unseen classes in Experiment A and overall best model for Experiment B are outlined in Table 3.  

| Model UID | Image Encoder Models | Text Encoders       | Training Datasets             | Experiment                     |
| --------- | -------------------- | ------------------- | ----------------------------- | ------------------------------ |
| 23        | CNN V2               | Bio_ClinicalBERT    | (PTB-XL, Ningbo, Georgia)     | Experiment A (Seen Classes)    |
| 20        | CNN base             | Bio_ClinicalBERT    | (PTB-XL, Ningbo, Georgia)     | Experiment A (Unseen Classes)  |
| 23        | CNN V2               | Bio_ClinicalBERT    | (PTB-XL, Ningbo, Georgia)     | Experiment B                   |

**Table 3.** Best CLIP-based in Experiment A and B.

| Model UID | PTB-XL Test Results | Ningbo Test Results | Georgia Test Results |
| --------- | ------------------- | ------------------- | -------------------- |
| 17        | 0.691±0.016         | -                   | -                    |
| 7         | 0.716±0.011         | 0.726±0.024         | -                    |
| 20        | 0.739±0.009         | 0.785±0.025         | 0.701±0.024          |
| 3         | 0.711±0.014         | -                   | -                    |
| 15        | 0.717±0.016         | 0.714±0.023         | -                    |
| 23        | 0.730±0.017         | 0.802±0.032         | 0.710±0.016          |
| 10        | 0.707±0.040         | -                   | -                    |
| 22        | 0.719±0.013         | 0.761±0.023         | -                    |
| 0         | 0.737±0.028         | 0.792±0.013         | 0.704±0.008          |
| 6         | 0.696±0.013         | -                   | -                    |
| 18        | 0.719±0.014         | 0.741±0.042         | -                    |
| 9         | 0.739±0.018         | 0.751±0.008         | 0.718±0.018          |
| 13        | 0.702±0.015         | -                   | -                    |
| 2         | 0.684±0.010         | -                   | -                    |
| 8         | 0.707±0.015         | 0.739±0.023         | -                    |
| 4         | 0.690±0.007         | 0.708±0.011         | -                    |
| 5         | 0.731±0.010         | 0.765±0.029         | 0.696±0.006          |
| 19        | 0.706±0.009         | 0.711±0.024         | 0.667±0.010          |
| 14        | 0.692±0.012         | -                   | -                    |
| 16        | 0.720±0.008         | 0.733±0.021         | -                    |
| 11        | 0.743±0.016         | 0.757±0.022         | 0.681±0.023          |
| 21        | 0.694±0.009         | -                   | -                    |
| 1         | 0.714±0.011         | 0.712±0.022         | -                    |
| 12        | 0.725±0.016         | 0.760±0.013         | 0.694±0.010          |

**Table 4.** Experiment A: Mean ROC-AUC scores with standard deviation for seen diagnostic classes in PTB-XL, Ningbo, and Georgia datasets. Fields denoted by "-" represents models for which mean ROC-AUC could not be computed due to these models not predicting at least one true positive sample for one or more ECG diagnostic classes.

| Model UID | PTB-XL Zero-Shot | Ningbo Zero-Shot | Georgia Zero-Shot |
| --------- | ----------------- | ----------------- | ------------------ |
| 17        | 0.563±0.037       | -                 | -                  |
| 7         | 0.613±0.017       | 0.641±0.014       | -                  |
| 20        | 0.632±0.024       | 0.668±0.032       | 0.698±0.016        |
| 3         | 0.541±0.021       | -                 | -                  |
| 15        | 0.610±0.027       | 0.601±0.004       | -                  |
| 23        | 0.615±0.035       | 0.640±0.015       | 0.698±0.009        |
| 10        | 0.544±0.059       | -                 | -                  |
| 22        | 0.599±0.017       | 0.646±0.023       | -                  |
| 0         | 0.638±0.021       | 0.661±0.012       | 0.694±0.005        |
| 6         | 0.596±0.019       | -                 | -                  |
| 18        | 0.606±0.032       | 0.613±0.021       | -                  |
| 9         | 0.641±0.046       | 0.604±0.019       | 0.676±0.030        |
| 13        | 0.590±0.048       | -                 | -                  |
| 2         | 0.565±0.006       | -                 | -                  |
| 8         | 0.619±0.011       | 0.626±0.017       | -                  |
| 4         | 0.583±0.013       | 0.580±0.010       | -                  |
| 5         | 0.624±0.033       | 0.623±0.024       | 0.686±0.018        |
| 19        | 0.573±0.026       | 0.598±0.034       | 0.649±0.006        |
| 14        | 0.587±0.011       | -                 | -                  |
| 16        | 0.600±0.039       | 0.551±0.034       | -                  |
| 11        | 0.611±0.040       | 0.570±0.011       | 0.633±0.016        |
| 21        | 0.585±0.024       | -                 | -                  |
| 1         | 0.589±0.033       | 0.582±0.036       | -                  |
| 12        | 0.623±0.020       | 0.616±0.030       | 0.686±0.005        |

**Table 5.** Experiment A: Mean ROC-AUC scores with standard deviation for unseen (zero-shot) diagnostic classes in PTB-XL, Ningbo, and Georgia datasets. Fields denoted by "-" represents models for which mean ROC-AUC could not be computed due to these models not predicting at least one true positive sample for one or more ECG diagnostic classes.

| Model UID | SPH Trained     | SPH Untrained   | CODE-15% Untrained |
| --------- | --------------- | --------------- | ------------------ |
| 17        | 0.698±0.034     | 0.568±0.043     | 0.578±0.042        |
| 7         | 0.786±0.028     | 0.666±0.019     | 0.623±0.017        |
| 20        | 0.817±0.016     | 0.665±0.028     | 0.636±0.019        |
| 3         | 0.717±0.038     | 0.577±0.025     | 0.540±0.024        |
| 15        | 0.784±0.038     | 0.670±0.038     | 0.608±0.029        |
| 23        | 0.829±0.029     | 0.703±0.014     | 0.637±0.028        |
| 10        | 0.703±0.081     | 0.585±0.091     | 0.579±0.064        |
| 22        | 0.805±0.027     | 0.668±0.019     | 0.606±0.034        |
| 0         | 0.817±0.024     | 0.682±0.029     | 0.634±0.024        |
| 6         | 0.721±0.039     | 0.593±0.043     | 0.580±0.022        |
| 18        | 0.783±0.029     | 0.637±0.044     | 0.596±0.025        |
| 9         | 0.817±0.014     | 0.679±0.030     | 0.659±0.030        |
| 13        | 0.735±0.017     | 0.607±0.044     | 0.600±0.043        |
| 2         | 0.693±0.029     | 0.578±0.027     | 0.549±0.016        |
| 8         | 0.772±0.022     | 0.669±0.022     | 0.590±0.027        |
| 4         | 0.725±0.014     | 0.616±0.007     | 0.571±0.014        |
| 5         | 0.799±0.017     | 0.679±0.015     | 0.608±0.018        |
| 19        | 0.732±0.033     | 0.621±0.004     | 0.583±0.024        |
| 14        | 0.710±0.045     | 0.570±0.030     | 0.581±0.036        |
| 16        | 0.785±0.009     | 0.619±0.033     | 0.574±0.048        |
| 11        | 0.812±0.017     | 0.619±0.020     | 0.625±0.018        |
| 21        | 0.705±0.017     | 0.582±0.023     | 0.585±0.021        |
| 1         | 0.773±0.011     | 0.636±0.010     | 0.602±0.016        |
| 12        | 0.804±0.006     | 0.683±0.019     | 0.629±0.027        |

**Table 6.** Experiment B: Mean ROC-AUC scores obtained with standard deviation for CLIP-based models retrained on SPH, untrained on SPH, and untrained of CODE-15% datasets.

## References

[1] Romera-Paredes, Bernardino, and Philip Torr. "An embarrassingly simple approach to zero-shot learning." International conference on machine learning. PMLR, 2015.

[2] Xian, Yongqin, et al. "Zero-shot learning—a comprehensive evaluation of the good, the bad and the ugly." IEEE transactions on pattern analysis and machine intelligence 41.9 (2018): 2251-2265.
