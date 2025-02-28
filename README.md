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

In ML/DL research, zero-shot learning refers to the problem setup of the experiments being conducted. In a zero-shot learning scenario, the main goal of the experiment is to evaluate a model's zero-shot classification performance. Zero-shot classification refers to the ability of a trained ML/DL model to identify and correctly classify new, unseen objects or information beyond the data models were trained on. One benefit of zero-shot learning is observed when working with multi-label datasets. In real world conditions, it is unlikely that every data sample will have an associated label or class to it. As such, training a ML/DL models with a zero-shot learning problem setup can be valuable in such cases as ML or DL models may be able to identify and correctly classify the unlabeled or unseen data. 

The zero-shot learning problem setup involves two main steps: model training and model inference. During the training phase, a ML or DL model is trained on a set of known classes from a dataset, where the model is expected to learn the representation of its inputs. Then, in the model inference phase, the trained model is evaluated on classes of data that were not included during the training phase, which in turn evaluates the model’s performance and ability to generalize to new, unseen classes of data.

In this study, CLIP-based models are trained using the zero-shot learning problem setup. It is hypothesized that CLIP-based models are well suited for zero-shot classification for two main reasons. First, a CLIP-based model consisting of an image and text encoder are jointly trained on ECG signals and their classes. This allows the CLIP-based model to learn deep semantic representations of the image-text pair inputs. Secondly, with a CLIP-based model, utilization of the contrastive loss function enables the model to effectively group similar pairs (positive instances) and dissimilar pairs (negative instances) of ECG image-text embeddings with distances of positive instances minimized and distances of negative instances maximized. 

In this study, CLIP-based models are trained on a specific set of ECG diagnostic classes. Then during evaluation, the trained CLIP-based models are tested on ECG diagnostic classes that were excluded during the training. The zero-shot learning problem setup allows us to evaluate the performance and ability of the CLIP-based model to generalize to new, unseen ECG diagnostic classes. One example of a real-world use case for such a CLIP-based model could be if the model were integrated with existing ECG signal software. For instance, given only an ECG signal and no other data, the CLIP-based model, if reasonably trained, can accurately provide a diagnosis given an ECG signal. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2cb2f4e8-a625-4003-8ddf-9a1fa5cf15dc" alt="White_ZeroShotLearning_ECGs drawio">
</p>
