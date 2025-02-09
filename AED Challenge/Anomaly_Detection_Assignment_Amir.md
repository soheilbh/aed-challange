## 1. **Contextual Anomaly Categorization**

- **Patient singing at 2 AM:** A **contextual anomaly** as singing is normal, but not at 2 AM in a sleeping room.

  - **Contextual attributes:** Time (2 AM), Location (sleeping room)
  - **Behavioral attribute:** Singing

- **Aggressive shouting:** A **point anomaly** as it stands out without needing contextual information.

- **May 4 commemoration of the dead:** A **contextual anomaly** due to the unusual sound pattern for that date. If it affects many rooms collectively, it may also be a **collective anomaly**.

## 2. **Technique Selection with Motivation**

- **Aggression detection from sound:**

  - **Nearest Neighbor** is preferred since normal speech forms dense clusters, while aggressive sounds stand apart.
  - Doesn't require labeled data, unlike classification.
  - Clustering and spectral methods are less suitable due to frequency variations and lack of distinct lower-dimensional structure.

- **Seizure detection from pressure sensors under the bed:**

  - **Classification** is best if labeled data is available, ensuring accuracy.
  - **Clustering** can be used if data is unlabeled, but may struggle with variations in seizure patterns.
  - Nearest Neighbor is prone to false positives due to normal movement noise.
  - Spectral techniques are less effective as seizures lack a consistent low-dimensional pattern.

- **Car/bike ball bearing monitoring via acceleration sensors:**
  - **Spectral techniques** are ideal due to high-dimensional vibration data and natural dimensionality reduction.
  - Nearest Neighbor is computationally expensive.
  - Clustering may misclassify minor anomalies.
  - Classification requires labeled data, which may not be available.

Real-world decisions would need a detailed system analysis, but these assumptions provide a reasonable basis for technique selection.
