### Read section 2.2 of Chandola et al. Answer the following question, in the context of the analysis of sound from say 250 sleeping rooms, monitored from a single location, what category of anomaly do the following situations constitute?

**Patient singing at 2 AM**: Since the patient sings at 2 AM (which is normal behavior at other times of the day but not at 2 AM midnight) in a sleeping room, it is a Contextual Anomaly (such as an investigation in a temperature time series where observing 35°F is normal in winter but abnormal in summer).

**Aggressive shouting**: This is a rare behavior (without considering time and location) compared to normal behavior, so it is a Point Anomaly (similar to credit card fraud detection mentioned in the paper).

**May 4 commemoration of the dead**: Since the behavior of a huge number of people changes on that night (not just one or a few individuals), it is a Collective Anomaly (such as human electrocardiogram output, where a low value exists for an abnormally long time). A collective anomaly occurs when a set of data (here sounds) deviates from a normal pattern as a group, even if each component alone is not considered an anomaly.



### If you had a contextual anomaly as category in the above question, then point out the contextual and behavioral attributes.
For the "Patient Singing at 2 AM":
- **Contextual Attributes**:
  - Time: 2 AM
  - Location: Sleeping room
- **Behavioral Attributes**:
  - Singing: Which is a normal behavior that becomes anomalous in this specific context due to it's specific contextual Attributes.

### For the following problems suggests with motivation, which technique from the sections above (classification, nearest neighbour, clustering or spectral) you would first try. Use the advantages and disadvantages mentioned in the paper to motivate your answer: 
**Aggression detection from sound**: 
- For this case, I would select the Nearest Neighbor method because of its assumption that normal data (normal speech, in this case) instances occur in dense neighborhoods, while anomalies (aggressive sounds) occur far from their closest neighbors. The aggressive sound is very different from other sounds in intensity and frequency, making it an outlier. 
- One of the advantages of this method is that Nearest Neighbor does not require labeled data, as preparing and labeling data is very difficult and expensive.
- The Clustering method seems inappropriate because of the different properties of aggressive noise (in frequency and amplitude), which may cause it not to be categorized in one cluster. 
- Spectral analysis would also not be my choice because of its assumption that "Data can be embedded into a lower-dimensional subspace in which normal instances and anomalies appear significantly different." Since aggressive sound instances do not necessarily have a clear lower-dimensional structure that separates them from normal speech or background noise, it also seems inappropriate.

**Seizure detection from pressure sensors under the bed.**: 
- If we consider that the pattern of seizures on the pressure sensors has been labeled (medical data is often labeled), classification would yield much better results compared to other methods. Classification is used to learn a model (classifier) from a set of labeled data instances (training) and then classify a test instance into one of the classes using the learned model."
- For unlabeled data, the Clustering technique would be my choice since it can handle data consisting of multiple normal patterns and works under the assumption that "Normal data instances belong to large and dense clusters, while anomalies either belong to small or sparse clusters. However, clustering would not work as well as classification due to the large variation within the same class of data (in this case, seizure events). 
- Spectral techniques seem inappropriate due to the lack of a unique pattern in pressure sensor data for normal and abnormal cases, making separation challenging in a reduced-dimensional space.
- Nearest Neighbor methods can be highly sensitive to noise and irrelevant features, leading to false positives and reduced accuracy in noisy datasets. Pressure sensor data naturally contains fluctuations due to movements unrelated to seizures, and Nearest Neighbor techniques may mistakenly classify these normal movements as seizures, leading to false alarms


**Car or bike ball bearing monitoring through acceleration sensors**:
- Spectral techniques automatically perform dimensionality reduction based on data embedded into a lower-dimensional subspace, and vibration data is high-dimensional, in which normal instances and anomalies appear significantly different. Thus, it would be my preferred choice for car or bike ball bearing monitoring through acceleration sensors.
- Since vibration data has a high-dimensional nature, Nearest Neighbor would be ineffective. 
- Clustering may not yield appropriate results since minor anomalies can still be assigned to normal clusters if large clusters are considered (one of the major disadvantages of clustering). 
- Classification, on the other hand, requires labeled data, which might not always be available, and preparing it could be very challenging and expensive.


 You need to speculate a bit about system behavior to make this work. In a real world assignment you would have to base your decision on thorough analysis of system behavior. Here you can work with what seem reasonable assumptions, you have to however make the assumptions underlying your reasoning explicit in your answer.
