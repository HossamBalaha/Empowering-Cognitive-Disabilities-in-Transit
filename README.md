# Empowering cognitive disabilities in transit: An explainable, emotion-aware ITS framework

## Overview

This repository contains the implementation of a novel framework that utilizes YOLOv8-based deep learning models to
recognize and interpret facial emotions in individuals with cognitive disabilities. The framework is designed to foster
better social integration by addressing the unique needs of this demographic.

### Key Features:

- **YOLOv8-based Deep Learning Models**: Advanced emotion detection.
- **EigenCam Explainability**: Intuitive visualizations of decision-making processes.
- **Adaptive Feedback Mechanisms**: Tailored interactions based on user cognitive profiles.
- **Integration with Assistive Technologies**: Support for augmented reality devices and mobile applications.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0
- Ultralytics YOLOv8

### Setup

1. Clone the repository:

```bash
   git clone https://github.com/HossamBalaha/Empowering-Cognitive-Disabilities-in-Transit.git
   cd Empowering-Cognitive-Disabilities-in-Transit
```

2. Install dependencies:

```bash
   pip install -r requirements.txt
```

## Description

The provided code encompasses several key components that contribute to the development and evaluation of a deep
learning-based framework for facial emotion recognition, specifically tailored for individuals with cognitive
disabilities. Below is a detailed description of the code in structured paragraphs:

The code begins by setting up the environment and preparing the dataset for training and evaluation. It utilizes the
splitfolders library to divide the dataset (e.g., CK+48) into training, validation, and testing subsets with a
predefined ratio (70% training, 15% validation, and 15% testing). This ensures a balanced distribution of data across
different phases of the machine learning pipeline. The dataset directory structure is organized such that each class (
emotion) has its own folder, and the splitting process maintains this structure. The input images are resized to a
uniform shape of 100x100 pixels to standardize the data for model training.

The core of the implementation revolves around training various YOLOv8 classification models (yolov8n, yolov8s, yolov8m,
yolov8l, yolov8x). These models are initialized using pre-trained weights (-cls.pt) and fine-tuned on the prepared
dataset. The training process involves specifying parameters such as the number of epochs (set to 250), image size, and
enabling options for saving plots and results. Each model variant is trained independently, and the performance
metrics (top-1 and top-5 accuracy) are logged after validation. The modular design allows for easy experimentation with
different model architectures and hyperparameters.

A critical component of the code is the CalculateMetrics function, which computes a comprehensive set of evaluation
metrics based on the confusion matrix derived from the model's predictions. These metrics include Accuracy, Precision,
Recall, Specificity, F1-score, Intersection over Union (IoU), Balanced Accuracy (BAC), and Matthews Correlation
Coefficient (MCC). The function handles multi-class classification scenarios and calculates weighted averages for each
metric, accounting for class imbalance in the dataset. This ensures that the evaluation reflects the model's performance
across all classes, providing a holistic view of its effectiveness.

The code aggregates results from multiple experiments by iterating through CSV files containing predictions and ground
truth labels for the test dataset. For each model variant, it reads the corresponding CSV file, extracts the actual and
predicted labels, and computes the evaluation metrics using the CalculateMetrics function. The results are compiled into
a Pandas DataFrame, which is then formatted into a LaTeX table for easy inclusion in academic publications. This
systematic approach to result aggregation facilitates comparative analysis of different model architectures and their
respective performances.

The code incorporates several utilities to ensure smooth execution. For instance, it addresses potential multiprocessing
issues by setting the KMP_DUPLICATE_LIB_OK environment variable to "TRUE," preventing conflicts related to OpenMP
libraries. Additionally, warnings are suppressed to maintain clean output during execution. These considerations reflect
attention to practical challenges that may arise during model development and evaluation.

## Materials

In this study, three publicly available datasets are utilized: CK+48, RAF-DB, and AffectNet. These datasets were
selected due to their relevance in facial emotion recognition research and their widespread use in benchmarking
approaches.

The CK+48 dataset, also known as the Extended Cohn-Kanade Dataset, is recognized as a benchmark in emotion recognition
studies. Each image sequence in CK+48 captures the transition from a neutral to a peak emotional expression. This
gradual progression facilitates robust model training by providing intermediate states that help in learning nuanced
facial transformations. The dataset includes frontal facial images with minimal occlusions and uniform lighting, which
simplifies preprocessing. Researchers have widely used CK+48 to evaluate models focused on distinguishing between
universally recognized emotions, making it a reliable choice for benchmarking the proposed approach.
Link: https://www.kaggle.com/datasets/gauravsharma99/ck48-5-emotions

RAF-DB extends the scope of facial emotion recognition by incorporating diverse real-world scenarios. The images in
RAF-DB are annotated using a crowdsourcing approach, ensuring high-quality and reliable labeling. The dataset's
inclusion of compound emotions (combinations of basic emotions) offers a more granular analysis of human emotional
states. This feature aligns with the need for advanced models capable of understanding subtle and mixed emotions.
Moreover, the diversity in demographics, poses, and lighting conditions in RAF-DB simulates real-world environments,
challenging the model to generalize effectively. Link: https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset

AffectNet significantly enhances the understanding of emotional expressions by providing a vast collection of images
annotated for both categorical and dimensional affective representations. Its categorical labels include emotions like
anger, disgust, fear, and happiness, while the dimensional labels provide arousal and valence scores, offering a richer
perspective on emotional intensity and polarity. The dataset includes challenging cases such as occlusions, extreme
poses, and images captured in varied cultural contexts. These attributes make AffectNet an essential resource for
training models that aim to achieve robust performance across diverse scenarios.
Link: http://mohammadmahoor.com/affectnet/

## Authors

**Malik Almaliki**
> College of Computer Science and Engineering, Taibah University
> King Salman Center for Disability Research


**Amna Bamaqa**
> Computer Science and Information Department, Applied College, Taibah University
> King Salman Center for Disability Research


**Tamer Ahmed Farrag**
> King Salman Center for Disability Research
> Department of Electrical Engineering, College of Engineering, Taif University


**Hossam Magdy Balaha**
> Faculty of Engineering, Computers and Control Systems Engineering Department, Mansoura University
> Bioengineering Department, University of Louisville


**Mahmoud Badawy**
> Computer Science and Information Department, Applied College, Taibah University
> Faculty of Engineering, Computers and Control Systems Engineering Department, Mansoura University


**Mostafa A. Elhosseini**
> College of Computer Science and Engineering, Taibah University
> Faculty of Engineering, Computers and Control Systems Engineering Department, Mansoura University

## Copyright and License

All rights reserved. No portion of this series may be reproduced, distributed, or transmitted in any form or by any
means, such as photocopying, recording, or other electronic or mechanical methods, without the express written consent
of the author. Exceptions are made for brief quotations included in critical reviews and certain other noncommercial
uses allowed under copyright law. For inquiries regarding permission, please contact the author directly. 
