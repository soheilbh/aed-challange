# Main Assignment Steps

## Understanding Your Data

- [x] read in only the wav files you want (10 classes from 5 different categories - you can use pandas to create a filtered list to get these, or use the end-number of each wav file to filter them when reading them in)
- [ ] visualise an instance of each class (as a spectogram or spectrum, for example)

## Extracting Features (*Individually Graded*)

- [ ] Extract lower dimensional features from your data (e.g. binned histograms from banded spectrograms or MFCCs)

- [ ] Create variations of these features (e.g. distribution of the bin edges, how you band the spectogram etc)

  {Advise: Save out the results to NPZ files so you don't have to regenerate these again}

## Modeling Your Data (Algorithm Development)

- [ ] Split your features into training and test sets (a ratio of 80:20 is recommended, but 90:10 also works)

- [ ] Split your training data into train and validation sets (you should use folds here)

- [ ] Fit each set of train data to your model and vary the model parameters:

  - [ ] For SVM (kernel type, C or Gamma)

  - [ ] For K-means (number of clusters) 

  - [ ] For others [Optional] e.g. K-medoids (convergence criteria)

    {Note: For unsupervised algorithms, you need to figure out how to apply labels to the clusters - maybe "majority values" or "closest data point"? 🤷🏽‍♂️}

- [ ] Use the validation data to assess the performance of said parameters (ROC and AUC may be used to select the best values)

- [ ] You should also end up with models based on different features 

- [ ] Select a subset of models based on performance (e.g. Linear SVM based on feature_set1, Linear SVM based on feature_set2, RBF SVM based on feature_set1, RBF SVM based on feature_set2, K-means with 10 clusters for feature_set2 etc etc )

## Comparing Your Models

- [ ] Use test set to calculate metrics such as accuracy, TPR, FPR
- [ ] Plot these out on a coverage plot
- [ ] Write a commetary on the performance of the models (based on the Coverage Plot, and maybe add a few confusion matrixes)