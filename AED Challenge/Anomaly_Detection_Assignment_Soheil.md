Patient singing at 2 AM would be a contextual anomaly. The context is time and location—2 AM in a sleeping room. The behavior is singing. Singing is normal, but in a sleeping room at 2 AM, it is not normal.

Aggressive shouting would be a point anomaly. It happens suddenly and is rare compared to normal sounds, so it stands out without needing context like time or place.

May 4 commemoration of the dead would be a collective anomaly. The context is the date, May 4, and the behavior is group noise or events. This would be unusual on a normal day, but on May 4, many people take part, making it collective.

For aggression detection from sound, I would use the nearest neighbor method. Aggressive sounds are very different from normal speech, so they stand out as outliers. Nearest neighbor methods do not require labeled data, which is an advantage when labeled training datasets are unavailable. However, they can be computationally expensive because they require calculating distances to multiple data points during testing (Chandola et al., Section 6).

For seizure detection from pressure sensors under the bed, classification would be the best choice if the data is labeled, while clustering would be used if the data is unlabeled. Classification-based techniques are good when accurate labels are available because they can use powerful algorithms to distinguish normal from anomalous instances. However, they require labeled data, which is often hard to get (Chandola et al., Section 3). Clustering techniques work in an unsupervised mode and are good for detecting anomalies without labels. The downside is that they often assign anomalies to large clusters, leading to missed anomalies (Chandola et al., Section 6).

For car or bike ball bearing monitoring, spectral analysis would be chosen. This method is suitable for high-dimensional data like vibration signals because it automatically reduces the data dimensions. However, spectral techniques only work well when normal and anomalous data are easily separable in the reduced space, which may not always be the case (Chandola et al., Section 4.3).

The assumptions are that aggressive sounds are different from normal sounds, pressure sensors can spot both normal and abnormal movements, and ball bearings normally have stable frequencies, with anomalies appearing when these frequencies change.

