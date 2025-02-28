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

In machine learning (ML) and deep learning (DL) research, zero-shot learning describes the experimental paradigm for evaluating a model's ability to perform classification on unseen classes. Specifically, zero-shot learning evaluates a model's zero-shot classification performance, the capacity to correctly classify data classes beyond those a model was trained on. This approach can be particularly beneficial in real-world scenarios where it is unlikely to have fully annotated, multi-label datasets. In short, training ML/DL models with a zero-shot learning paradign can improve the potential of models to classify unseen and unlabeled data, both of which are inherent limitations of real-world datasets. 

The zero-shot learning paradigm involves two key steps: (1) model training and (2) model inference. During model training, ML/DL models are trained on a defined set of known classes where models are expected to learn feature representations of their inputs. Then, during model inference, these models are evaluated for their performance on classes excluded during training to measure classification and generalization performance on new, unseen classes. 

In this study, Contrastive Language–Image Pre-training (CLIP) based models are trained using the zero-shot learning framework. It is hypothesized that CLIP-based models are well suited for zero-shot learning for two key reasons. First, the joint training of image and text encoders enables better learning of the semantic relationships of ECG image-text pair inputs. Second, the contrastive loss function inherent to CLIP allows for effective discrimination between similar (positive instances) and dissimilar (negative instances) pairs of ECG image-tet embeddings, minimizing the distance between positive instances while maximizing the distance between negative instances.    

In this study, CLIP-based models are trained on a specific set of ECG diagnostic classes. The classification performane of these models are evaluated on a set of ECG diagnostic classes that were excluded during the training. The zero-shot learning framework allows us to evaluate the performance and ability of the CLIP-based model to generalize to new, unseen ECG diagnostic classes. One potential real-world application of an ECG CLIP-based model includes the integration of said model with existing ECG analysis software. For example, given only an ECG signal as input, a reasonable trained CLIP-based model could provide accurate diagnostic predictions. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2cb2f4e8-a625-4003-8ddf-9a1fa5cf15dc" alt="White_ZeroShotLearning_ECGs drawio">
  <br>
  Figure 1. Zero-shot learning for ECG classification.
</p>
