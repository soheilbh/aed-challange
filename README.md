[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/wvQEO1LN)

---

# Acoustic Event Detection Challenge (AED)

### Main Assignment for Applied Machine Learning

This repository contains our **Acoustic Event Detection Challenge** project, part of an Applied Machine Learning course. The central theme is analyzing audio data from the **ESC-50** dataset, extracting meaningful features, and applying machine learning algorithms (both supervised and unsupervised) to classify environmental sounds effectively.

---

#### ✨ Live Demo

We developed a **live demo service** where you can try **SVM and model training** with different settings and test with sounds online. Check it out here: [Sound Detection Demo](https://sound-detection.zal.digital/)

---

## Table of Contents

1. [Overview & Assignment Description](#overview--assignment-description)
2. [Main Assignment Steps](#main-assignment-steps)
   - [1. Understanding Your Data](#1-understanding-your-data)
   - [2. Extracting Features (Individually Graded)](#2-extracting-features-individually-graded)
   - [3. Modeling Your Data (Algorithm Development)](#3-modeling-your-data-algorithm-development)
     - [3.1 Support Vector Machine (SVM)](#31-support-vector-machine-svm)
     - [3.2 Random Forest Classifier](#32-random-forest-classifier)
     - [3.3 K-Nearest Neighbors (KNN)](#33-k-nearest-neighbors-knn)
     - [3.4 Unsupervised Learning - K-Means Clustering](#34-unsupervised-learning---k-means-clustering)
   - [4. Comparing Your Models](#4-comparing-your-models)
3. [Repository Structure](#repository-structure)
4. [Installation & Requirements](#installation--requirements)
5. [Feature Extraction Contributions](#feature-extraction-contributions)
   - [Amir's Feature Extraction](#amirs-feature-extraction)
   - [Mina's Feature Extraction](#minas-feature-extraction)
   - [Soheil's Feature Extraction](#soheils-feature-extraction)
6. [PCA & ICA Notebooks](#pca--ica-notebooks)
   - [Amir's Notebook](#amirs-notebook)
   - [Mina's Notebook](#minas-notebook)
   - [Soheil's Notebook](#soheils-notebook)
7. [Team & Contributions](#team--contributions)

---

## Overview & Assignment Description

### The Acoustic Event Detection Challenge

In many nursing homes, acoustic monitoring is used to detect potential alarms or anomalies. However, simple detection methods based on **sound intensity** and **duration** often result in high false-alarm rates, while some short-lived but important events may be missed.

**Goal**:

- Develop machine learning solutions that **classify** or **detect** specific acoustic events.
- Use the **ESC-50** dataset (environmental sound recordings) to explore data from **10 classes** chosen from **5 different categories**.

The **main research question** guiding this project is:

> **Can we infer the source of a sound recording from extracted features?**

**Subquestions** include:

1. How can we define a suitable set of features?
2. Which algorithms can be used to solve this task?
3. How does performance depend on algorithm choice and their parameters?
4. How does performance depend on feature extraction parameters?
5. How well do supervised algorithms perform against unsupervised algorithms for the task?

---

[See K-Means Clustering](#34-unsupervised-learning---k-means-clustering)

### 4. Comparing Your Models

[See Comparing Your Models](#4-comparing-your-models)

## Overview & Assignment Description

### The Acoustic Event Detection Challenge

In many nursing homes, acoustic monitoring is used to detect potential alarms or anomalies. However, simple detection methods based on **sound intensity** and **duration** often result in high false-alarm rates, while some short-lived but important events may be missed.

**Goal**:

- Develop machine learning solutions that **classify** or **detect** specific acoustic events.
- Use the **ESC-50** dataset (environmental sound recordings) to explore data from **10 classes** chosen from **5 different categories**.

The **main research question** guiding this project is:

> **Can we infer the source of a sound recording from extracted features?**

**Subquestions** include:

1. How can we define a suitable set of features?
2. Which algorithms can be used to solve this task?
3. How does performance depend on algorithm choice and their parameters?
4. How does performance depend on feature extraction parameters?
5. How well do supervised algorithms perform against unsupervised algorithms for the task?

---

### 1. Understanding The Data

We selected 10 classes from five categories in the ESC-50 dataset:

- **Animals**: Dog, Rooster
- **Natural Sounds**: Thunderstorm, Pouring Water
- **Human Sounds**: Snoring, Sneezing
- **Interior Sounds**: Clock Alarm, Vacuum Cleaner
- **Urban Noises**: Siren, Helicopter

#### **Data Filtering & Class Balance**

We filtered the dataset to retain only the selected classes and checked for class balance. Each class contained 40 samples, ensuring uniform distribution.

#### **Visualizing Category Distribution**

A pie chart was plotted to illustrate the proportion of each class in the dataset.

#### **Mapping Categories to Target Labels**

We mapped each category to its respective target label, as shown below:

| Category       | Target |
| -------------- | ------ |
| Dog            | 0      |
| Rooster        | 1      |
| Thunderstorm   | 19     |
| Pouring Water  | 17     |
| Clock Alarm    | 37     |
| Vacuum Cleaner | 36     |
| Helicopter     | 40     |
| Siren          | 42     |
| Snoring        | 28     |
| Sneezing       | 21     |

#### **Loading Audio Data**

We loaded the `.wav` files, obtaining:

- **400 total samples**
- **Sample rate:** 44,100 Hz
- **Waveform shape:** (220,500)

This prepared the dataset for feature extraction and model training.

##### More Visualizations => [Data Visualizations](Data_Visualization.ipynb)

### 2. Extracting Features

#### **Key Features and Their Relevance**

We extracted a diverse set of audio features, each contributing uniquely to sound classification. Below is a breakdown of these features and their importance:

1. **MFCC (Mel Frequency Cepstral Coefficients)**

   - **Description:** Encodes spectral shape and helps distinguish tonal variations.
   - **Best for:** Animal sounds (e.g., dog, rooster) and human speech.
   - **Why Include?** Captures timbre effectively, essential for differentiating speech-like sounds.

2. **Delta MFCC (Temporal Changes in MFCC)**

   - **Description:** Measures how MFCCs change over time, useful for dynamic audio patterns.
   - **Best for:** Snoring, thunderstorms, and vacuum cleaners.
   - **Why Include?** Helps differentiate steady vs. time-varying sounds.

3. **Spectral Centroid**

   - **Description:** Indicates the center of mass of the spectrum, relating to sound brightness.
   - **Best for:** High-pitched or harmonic sounds like sirens and alarms.
   - **Why Include?** Distinguishes sharp, high-frequency sounds from lower, dull ones.

4. **Spectral Contrast**

   - **Description:** Measures the difference between spectral peaks and valleys.
   - **Best for:** Helicopter, sirens, and natural sounds with background noise.
   - **Why Include?** Helps differentiate between structured and unstructured harmonic sounds.

5. **Pitch Features**

   - **Description:** Extracts fundamental frequency and harmonic components.
   - **Best for:** Animal sounds like dog barking and rooster calls.
   - **Why Include?** Effectively separates tonal sounds from atonal noise.

6. **Zero-Crossing Rate (ZCR)**

   - **Description:** Counts the rate of amplitude sign changes in a waveform.
   - **Best for:** Short, impulsive sounds like sneezing and clock alarms.
   - **Why Include?** High ZCR indicates transient, percussive sounds.

7. **Envelope (Amplitude Envelope)**

   - **Description:** Tracks amplitude changes over time.
   - **Best for:** Sounds with fluctuating intensity, such as thunderstorms and alarms.
   - **Why Include?** Captures loudness variations, useful for classifying dynamic sounds.

8. **Harmonic-to-Noise Ratio (HNR)**
   - **Description:** Measures harmonic content relative to noise.
   - **Best for:** Noisy sounds (e.g., vacuum cleaner) vs. harmonic sounds (e.g., rooster).
   - **Why Include?** Differentiates noise-dominant signals from tonal ones.

#### **Feature Selection Recommendations**

To optimize classification performance, we prioritized features based on sound type:

- **Animal Sounds:** MFCC, Delta MFCC, Pitch Features, HNR
- **Natural Soundscapes:** Spectral Centroid, Spectral Contrast, Envelope, MFCC
- **Human Sounds:** MFCC, Delta MFCC, ZCR, Pitch Features
- **Interior/Domestic Sounds:** HNR, Spectral Contrast, ZCR
- **Urban/Exterior Noises:** Spectral Centroid, Spectral Contrast, HNR

#### **Optimization Strategy**

1. **Initial Feature Combinations:**

   - Tested K-Means/K-Medoids with various feature subsets (MFCC, Delta MFCC, Spectral Contrast).
   - Evaluated performance using **Silhouette Score, Adjusted Rand Index (ARI), and Majority Class Proportion (MCP).**

2. **Dimensionality Reduction:**

   - Applied **PCA** to reduce noise while preserving meaningful feature variations.

3. **Majority Voting for Label Assignment:**
   - After clustering, assigned labels based on the majority class within each cluster.

#### **Next Steps**

- Fine-tune feature selection based on **Silhouette Score, ARI, and MCP.**
- Experiment with alternative clustering algorithms if K-Means/K-Medoids underperform.
- Normalize all features before clustering for better accuracy.
- Validate feature selection using **t-SNE visualizations and confusion matrices.**

By refining feature extraction and selection, we ensure robust classification performance for environmental sounds.

### 3. Modeling The Data (Algorithm Development)

After extracting and selecting relevant features, we trained classification models to distinguish environmental sound categories. Our approach involved feature group selection, data normalization, training machine learning models, and evaluating their performance.

---

#### **Feature Group Selection**

We defined **seven feature groups** based on different sound properties:

1. **Temporal Features:** Delta MFCC, Zero Crossing Rate (ZCR), and Envelope.
2. **Harmonic Features:** MFCC, Pitch Features, and Harmonic-to-Noise Ratio (HNR).
3. **Spectral Brightness:** MFCC, Spectral Centroid, and Spectral Contrast.
4. **Noise-Based Features:** Histogram, Spectral Contrast, and HNR.
5. **General Set 1:** MFCC, Delta MFCC, ZCR, and Envelope.
6. **General Set 2:** MFCC, Spectral Centroid, Spectral Contrast, and Pitch Features.
7. **Composite Model:** A combination of all the above.

---

#### **Data Preparation & Normalization**

1. **Train-Test Split (80-20 Ratio with Stratification):**

   - Ensured that each class had proportional representation in the train and test sets.
   - **Training Set Size:** 320 samples
   - **Test Set Size:** 80 samples

2. **Feature Scaling (Z-Score Normalization):**
   - Used **StandardScaler** to normalize feature distributions.
   - Applied **mean and standard deviation from training data** to both train and test sets.

---

### 3.1 Support Vector Machine (SVM)

1. **Model Selection:**

   - Chose an **SVM with an RBF kernel** to capture non-linear feature interactions.
   - Hyperparameters:
     - **Kernel:** RBF
     - **C:** 10
     - **Gamma:** Scale

2. **Model Training & Predictions:**

   - Trained the model on normalized training features.
   - Predicted class labels and probabilities on the test set.

3. **Performance Metrics:**
   - **AUC (Area Under Curve - Multi-Class):** Evaluated how well the model separates classes.
   - **Classification Report:** Provided precision, recall, and F1-score for each class.
   - **Confusion Matrix & Accuracy:** Analyzed misclassifications and overall correctness.

---

#### **Results & Insights**

- **AUC Score:** Achieved a strong multi-class separation.
- **Accuracy:** Provided a reliable prediction rate across all sound categories.
- **Confusion Matrix:** Identified misclassified instances, showing similarities between some sound types.

---

We trained and evaluated the SVM classifier using different feature combinations, cross-validation techniques, and performance metrics.

---

#### **Feature Selection & Combination Analysis**

1. **Top 5 Non-Overfitted Feature Combinations:**

   - Filtered **feature sets with exactly 3 features** that showed no overfitting.
   - Sorted by **combined score** (AUC & validation accuracy).
   - Visualized results using a **bar & line plot** to compare **accuracy and AUC scores** across the top 5 combinations.

2. **Final Feature Selection:**
   - Chose a **diverse mix of spectral, harmonic, and temporal features** to balance representation:
     - **MFCC, Delta MFCC, Spectral Centroid, Spectral Contrast, Pitch Features, ZCR, Envelope, HNR**
   - **Feature space:** **54-dimensional** representation after feature expansion.

---

#### **Support Vector Machine (SVM) Training & Evaluation**

1. **Data Splitting:**

   - **80-20 Train-Test Split** with **stratification** to maintain class distribution.

2. **Normalization & Cross-Validation:**

   - Applied **Z-score normalization** to standardize feature distributions.
   - Used **5-Fold Cross-Validation** to:
     - Evaluate model performance on multiple subsets.
     - Detect potential **overfitting** using a **threshold of 0.1**.

3. **Model Configuration:**
   - **Kernel:** Radial Basis Function (RBF)
   - **C:** 1.0
   - **Gamma:** 0.01
   - **Probability Estimates:** Enabled for multi-class probability outputs

---

#### **Performance Metrics**

#### **SVM Classification Report (Last Fold of Cross-Validation):**

- **Accuracy:** **84%** on the test set.
- **Precision & Recall:** High performance across most classes, with a **macro average of 85% precision and 84% recall**.
- **Best Performing Classes:** Rooster, Pouring Water, Sneezing, and Siren.
- **Areas of Improvement:** Slight confusion between similar classes (e.g., Helicopter vs. Vacuum Cleaner).

#### **Confusion Matrix Analysis**

- The **confusion matrix** was plotted to analyze misclassifications.
- Most misclassifications occurred between **acoustically similar classes**, such as:
  - Vacuum Cleaner ↔ Helicopter
  - Clock Alarm ↔ Siren

---

#### **Insights & Next Steps**

- **Feature Optimization:** Further refine top **3-feature combinations** for better generalization.
- **Dimensionality Reduction:** **Apply PCA** to explore lower-dimensional embeddings for performance gains.
- **Model Comparison:** Evaluate additional classifiers (e.g., **Random Forest, KNN**) to benchmark against SVM.
- **Hyperparameter Tuning:** Fine-tune **C & Gamma** for potential improvements.

This robust modeling approach ensures a **balanced and high-performing classification system** for environmental sounds.

### 3.2 Random Forest Classifier

We evaluated **Random Forest (RF)** as an alternative to SVM for environmental sound classification.

### **Hyperparameter Tuning & Training**

1. **Stratified K-Fold Cross-Validation (5-Fold)**

   - Ensured balanced class distribution across folds.

2. **Hyperparameter Optimization (GridSearchCV)**

   - Explored different **tree depths, number of estimators, and split criteria**.
   - **Best Parameters:**
     - **Max Depth:** 10
     - **Min Samples Split:** 3
     - **Min Samples Leaf:** 2
     - **Number of Trees:** 100

3. **Final Model Training**
   - Trained **RandomForestClassifier** with optimal parameters.

---

### **Evaluation Metrics**

- **AUC Score (Test Set):** **0.993**
- **Accuracy (Test Set):** **88.75%**

- **Top Performing Classes:**
  - Dog, Rooster, Snoring, Siren (Precision & Recall ~ 1.00)
- **Areas for Improvement:**
  - Helicopter & Clock Alarm had lower recall (~ 0.62 - 0.75).

---

### **Random Forest - Final Evaluation**

#### **Performance Metrics**

- **AUC Score (Test Set):** **0.993**
- **Accuracy (Test Set):** **91.25%**

#### **Classification Report Insights**

- **High Precision & Recall:** Siren, Clock Alarm, and Snoring (F1-score ~ 1.00).
- **Moderate Performance:** Helicopter and Pouring Water had slightly lower recall (~ 0.75).

#### **Confusion Matrix Analysis**

- Most classes were classified correctly with minimal misclassifications.
- Minor overlaps between **acoustically similar sounds** like **Helicopter & Clock Alarm**.

---

### 3.3 K-Nearest Neighbors (KNN)

We trained and optimized **KNN** for environmental sound classification using cross-validation and feature selection.

---

#### **Model Training & Hyperparameter Tuning**

1. **Feature Selection & Preprocessing:**

   - Selected key features (**MFCC, Delta MFCC, Spectral Centroid, Spectral Contrast, etc.**)
   - Applied **Z-score normalization**, PCA disabled.

2. **Initial KNN Training:**

   - Used **k=3** neighbors.
   - **Test Accuracy:** **70%** | **AUC:** **0.906**
   - Indications of **underfitting** in some classes.

3. **Hyperparameter Optimization (Grid Search):**
   - **Best Parameters:**
     - **k=7**, **uniform weights**, **Euclidean distance**
   - Improved overall performance.

---

#### **Final KNN Evaluation (Optimized Model)**

- **Test Accuracy:** **73.75%**
- **AUC Score:** **0.962**
- **Training Accuracy:** **80.94%** (Minimal overfitting)

#### **Classification Report Insights:**

- **Strong Performance:** Rooster, Snoring, Siren (**F1-score > 0.80**)
- **Lower Recall:** Pouring Water, Helicopter (**0.38 - 0.50**)

#### **Confusion Matrix Analysis:**

- Some **misclassifications between similar sounds** (e.g., **Helicopter ↔ Clock Alarm**).

---

### 3.4 Unsupervised Learning - K-Means Clustering

Since our dataset already had **10 labeled classes**, clustering was a **challenging task**. However, we applied **K-Means** to explore **natural groupings** and assess its performance against known labels.

---

#### **Feature Selection & Preprocessing**

- Used **general feature group (`group_6_general_2`)** for clustering.
- Combined **training & test data** to create a full dataset.
- Applied **Z-score normalization** and **PCA (95% variance retained).**

---

### **Finding the Optimal Number of Clusters (`k`)**

- **Elbow Method:** Plotted **inertia (sum of squared distances)** to identify the best `k`.
- **Silhouette Score:** Measured clustering quality across different `k` values.
- Best performance observed near **k = 10**, aligning with the dataset structure.

---

### **Clustering Evaluation (`k=10`)**

1. **K-Means Clustering (k=10, K-Means++ Initialization)**

   - **Silhouette Score:** **0.42** (Moderate clustering quality)
   - **ARI Score:** **0.35** (Moderate alignment with true labels)

2. **t-SNE Visualization:**

   - Plotted **2D representation** of clusters, revealing overlapping groups.

3. **Assigning Labels to Clusters (Majority Voting)**

   - **Mapped clusters to known classes** by assigning the **most frequent true label** per cluster.

4. **Confusion Matrix Analysis:**

   - Clusters **partially aligned** with true classes but **showed significant overlap**.

5. **Cluster Size Distribution:**
   - Some clusters were **dominant**, while others had **imbalanced distributions**.

---

### **Key Takeaways & Challenges**

- **Clustering did not perfectly match labels**, as expected.
- **Overlap among acoustically similar sounds** (e.g., vacuum cleaner vs. helicopter).
- **Future Improvements:**
  - Try **Gaussian Mixture Models (GMM)** for more flexible clusters.
  - Explore **DBSCAN or Agglomerative Clustering** for alternative structures.
  - Incorporate **self-supervised learning** to enhance representations.

While **unsupervised learning struggled with known class boundaries**, it provided **insightful groupings** and **potential feature refinements** for future analysis.

### 4. Comparing The Models

We compared the performance of **KNN, SVM, and Random Forest** using various visualization techniques:

1. **Bar and Line Plot:** Showed **Test Accuracy, TPR, FPR, and AUC** across models. **SVM performed the best with 85% accuracy**, followed by **Random Forest at 78%**, and **KNN at 76%**.
2. **Radar Chart:** Provided a **holistic performance comparison** across multiple metrics like **Train Accuracy, Test Accuracy, AUC, and False Positive Rate (FPR)**.
3. **Key Takeaways:**
   - **SVM achieved the best balance of accuracy, precision, and recall.**
   - **Random Forest had high train accuracy but signs of overfitting.**
   - **KNN performed the worst overall**, making it less suitable for this dataset.
   - **All models maintained a very low False Positive Rate (FPR).**
   - **If recall is a priority, SVM is the best choice.**
   - **For balancing accuracy and precision, both SVM and Random Forest are strong options.**

---

## Repository Structure

```plaintext
📂 AED Challenge
│   └── Anomaly_Detection_Assignment-Mina.md

📂 Anomaly Detection
│   └── .gitkeep

📂 ESC-50-master
│   └── ... (audio dataset files and structure)

📂 features
│   ├── .gitkeep
│   ├── extracted_features_multiple_test.npz
│   ├── extracted_features_test.npz
│   └── extracted_features.npz

📂 PCA_ICA
│   ├── __pycache__/
│   ├── dataset/
│   │   ├── .gitkeep
│   │   ├── DejaVuSerif.ttf
│   │   └── fontdemo.py
│   ├── pca_ica.ipynb             (Amir notebook)
│   ├── PCA-ICA_Mina.ipynb        (Mina's notebook)
│   ├── PCA-ICA_Soheil.ipynb      (Soheil's notebook)
│   ├── .gitignore
│   ├── data_utils.py
│   ├── feature_utils.py
│   ├── Main Assignment Steps 2024.md
│   ├── main.ipynb
│   ├── plot_utils.py
│   └── README.md                 (You are here)
```

- **AED Challenge**: Contains anomaly detection assignment files.
- **Anomaly Detection**:
- **ESC-50-master**: Dataset folder with .wav files for 50 classes. We focus on 10 classes.
- **features**: Stores extracted feature files (`.npz`).
- **PCA_ICA**:
  - **Notebooks** for PCA/ICA workflows.
  - **Utility scripts** for data loading, feature engineering, and plotting.

---

## Installation & Requirements

1. **Clone** the repository:
   ```bash
   git clone https://github.com/HanzeUAS-MasterSSE/AML-Soheil-Mina-Amir.git
   ```
2. **Install dependencies** (in a virtual environment recommended):

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not provided, manually install packages mentioned in the notebooks.

3. Ensure you have the correct **ESC-50** .wav files in the **ESC-50-master** folder or link them properly.

---

## Feature Extraction Contributions

Each team member (Amir, Mina, Soheil) explored feature extraction differently. Below is a summary;

### Amir's Feature Extraction

[Your Explanation Here]

### Mina's Feature Extraction

[Your Explanation Here]

### Soheil's Feature Extraction

[Your Explanation Here]

---

## PCA & ICA Notebooks

Detailed dimensionality-reduction approaches and the final extracted components can be found in each member’s notebook:

##### Refer to the following each member's notebook for detailed PCA/ICA analysis:

### Amir's Notebook

- **File**: [PCA-ICA_Amir.ipynb](PCA_ICA/pca_ica.ipynb)
- Words Founded: - `Fink` - ``

### Mina's Notebook

- **File**: [PCA-ICA_Mina.ipynb](PCA_ICA/PCA-ICA_Mina.ipynb)
- Words Founded: - `word here` - `word here`

### Soheil's Notebook

- **File**: [PCA-ICA_Soheil.ipynb](PCA_ICA/PCA-ICA_Soheil.ipynb)
- Words Founded: - `word here` - `word here`

---

## Contributions

- **Mina**
- **Soheil**
- **Amir**

Each member contributed to feature extraction, PCA/ICA analysis, and the overall architecture of the project.

---

<p align="center"> Crafted by Mina, Soheil & Amir </p>
