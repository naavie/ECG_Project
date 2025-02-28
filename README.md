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
