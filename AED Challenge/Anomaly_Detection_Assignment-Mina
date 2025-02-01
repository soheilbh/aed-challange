### Read section 2.2 of Chandola et al. Answer the following question, in the context of the analysis of sound from say 250 sleeping rooms, monitored from a single location, what category of anomaly do the following situations constitute?

**Patient singing at 2 AM**: Since the patient sings at 2 AM (not at other times of the day) in a sleeping room, it is a "Contextual Anomaly" (such as an investigation in a temperature time series and observing 35°F being normal in winter but abnormal in summer).

**Aggressive shouting**: This is a rare behavior (without time and location consideration) compared to normal behavior, so it is a "Point Anomaly" (such as credit card fraud detection).

**May 4 commemoration of the dead**: Since the behavior of a huge number of people changes on that night (not only one or a few of them), it is a Collective Anomaly (such as human electrocardiogram output in which low value exists for an abnormally long time).


### If you had a contextual anomaly as category in the above question, then point out the contextual and behavioral attributes.
For the "Patient Singing at 2 AM":
- **Contextual Attributes**:
  - Time: 2 AM
  - Location: Sleeping room
- **Behavioral Attributes**:
  - Singing: Which is a normal behavior that becomes anomalous in this specific context due to it's specific contextual Attributes.

### For the following problems suggests with motivation, which technique from the sections above (classification, nearest neighbour, clustering or spectral) you would first try. Use the advantages and disadvantages mentioned in the paper to motivate your answer: 
**Aggression detection from sound**: For this case, I would select the Nearest Neighbor method because of the assumption mentioned in this method: "Normal data instances occur in dense neighborhoods, while anomalies occur far from their closest neighbors." The aggressive sound is very different from other sounds in intensity and frequency, making it an outlier. One of the advantages of this method is that Nearest Neighbor does not require labeled data, because preparing and labeling data is very difficult and expensive. The other methood which could be very efficient is SVM. "For each test instance, the basic technique determines if the test instance falls within the learned region (normal speech). If a test instance falls within the learned region, it is declared as normal, else it is declared as anomalous."
Clustering method seems to be inappropriate because of the different properties of aggressive noise, which may cause it not to be categorized in one cluster. Spectral analysis also would not be my choice, becasuse of it's assumtion; "Data can be embedded into a lower dimensional subspace in which normal instances and anomalies appear significantly different." and since aggressive sounds instances do not necessarily have a clear lower-dimensional structure that separates them from normal speech or background noise, so it also seems to be inappropriate.
**Seizure detection from pressure sensors under the bed.**: The Spectral techniques seems to be inappropriate due to lack of one unique pattern in data of the pressure sensors in normal situation, making the seperation challenging. On the other hand, If we consider that the pattern of Seizure on the pressure sensors has been labeled (medical data is often labeled), so classification would have much better result compared to others. "Classification is used to learn a model (classifier) from a set of labeled data instances (training) and then, classify a test instance into one of the classes using the learned model (testing)". While, for data without labling, Clustering technique would be my choice, since it can handle data consisting of multiple normal patterns and works with this assumption; "Normal data instances belong to large and dense clusters, while anomalies either belong to small or sparse clusters."
**Car or bike ball bearing monitoring through acceleration sensors**:Spectral techniques automatically perform dimensionality reduction based of data embedded into a lower dimensional subspace in which normal instances and anomalies appear significantly different, so it would be my choice for car or bike ball bearing monitoring through acceleration sensors. 


 You need to speculate a bit about system behavior to make this work. In a real world assignment you would have to base your decision on thorough analysis of system behavior. Here you can work with what seem reasonable assumptions, you have to however make the assumptions underlying your reasoning explicit in your answer.
