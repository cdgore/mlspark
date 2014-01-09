mlspark
=======
[![Build Status](https://travis-ci.org/cdgore/mlspark.png?branch=master)](https://travis-ci.org/cdgore/mlspark)

Machine learning algorithms implemented in Scala on Spark

Currently 4 models are included:

* **Gaussian Naive Bayes** – Naive Bayes classifier for continuous features.  Assumes likelihoods follow Gaussian distribution P(_x\_i_ | _y_) = (1/sqrt(2 \* _pi_ \* _sigma\_y_^2)) \* exp(-((_x\_i_ - _mu\_y_)^2)/2 \* _pi_ \* _sigma\_y_^2).  The posterior distribution for each class is estimated by summing the exponential of all likelihoods and for a given class and class prior probability.
* **K Means** – Performs k-means clustering on data samples labeled by class.  The distance function ```distMeasure``` may be specified as either ```euclidean``` (default) or ```cosine```.  Distance functions are passed internally as partially defined functions for extensibility.  Both the means and standard deviations are calculated and recorded for each cluster - useful for generating radial basis functions based on distance from clusters.
* **Logistic Regression** – Binary logistic regression classifier with L2 normalization.  Loss function is minimized with gradient descent
* **Softmax Logistic Regression** – Multi-class logistic regression with optional regularizations: L1, L1 (with clipping), L2, none (default).  Regularization gradient update functions are specified and passed as partials for extensibility.