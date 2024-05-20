# SPAM-SMS-DETECTION

problem statment:
Build an AI model that can classify SMS messages as spam or legitimate. Use techniques like TFIDF or word embeddings with classifiers like Naive Bayes, Logistic Regression, or Support Vector Machines to identify spam messages.Additionally, Imagine you've been tasked with creating comprehensive documentation for a SPAM SMS DETECTIONproject.Your documentation should include detailed explanations of the code along with images illustrating the program's execution and user interactions. 

**ABOUT DATA SET**:

***Context***

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

***Content***

The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.
This corpus has been collected from free or free for research sources at the Internet:
-> A collection of 425 SMS spam messages was manually extracted from the Grumbletext Web site. This is a UK forum in which cell phone users make public claims about SMS spam messages, most of them without reporting the very spam message received. The identification of the text of spam messages in the claims is a very hard and time-consuming task, and it involved carefully scanning hundreds of web pages. 
-> A subset of 3,375 SMS randomly chosen ham messages of the NUS SMS Corpus (NSC), which is a dataset of about 10,000 legitimate messages collected for research at the Department of Computer Science at the National University of Singapore. The messages largely originate from Singaporeans and mostly from students attending the University. These messages were collected from volunteers who were made aware that their contributions were going to be made publicly available. The NUS SMS Corpus is avalaible at: [Web Link].
-> A list of 450 SMS ham messages collected from Caroline Tag's PhD Thesis available at [Web Link].
-> Finally, we have incorporated the SMS Spam Corpus v.0.1 Big. It has 1,002 SMS ham messages and 322 spam messages and it is public available at: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data.

**Data Collection and Preprocessing:**

-Gather a labeled dataset of SMS messages, where each message is tagged as “spam” or “ham” (legitimate).
-Preprocess the text data by removing stop words, special characters, and converting everything to lowercase.
-Split the dataset into training and testing subsets.

Feature Extraction:

-Use techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings (such as Word2Vec or GloVe) to represent the SMS messages as numerical features.
-TF-IDF captures the importance of each word in a document relative to the entire corpus.
-Word embeddings create dense vector representations for words based on their context.

**Model Selection and Training:**

Choose a classifier. Some popular choices include

**Logistic Regression**

Logistic Regression models the relationship between the independent variables (features) and the probability of a particular outcome using the logistic function (also known as the sigmoid function).The hypothesis function in logistic regression is defined as the sigmoid function applied to the linear combination of the input features and their corresponding weights.The decision boundary in logistic regression separates the classes in the feature space and is represented by a linear equation.The cost function for logistic regression, known as the logistic loss or cross-entropy loss, penalizes incorrect predictions by measuring the difference between the predicted probabilities and the actual labels.Parameters (coefficients) in logistic regression are estimated using optimization algorithms such as gradient descent or Newton's method to minimize the cost function.Regularization techniques like L1 or L2 regularization can be applied to logistic regression to prevent overfitting.Evaluation metrics for logistic regression include accuracy, precision, recall, F1-score, and ROC curve.Logistic Regression is widely used in various domains such as healthcare (disease prediction), finance (credit risk assessment), marketing (customer churn prediction), and more.Assumptions of logistic regression include linearity between independent variables and the log-odds of the dependent variable.Pros of logistic regression include its simplicity, interpretability, efficiency in training and prediction, and suitability for small datasets and linearly separable problems.Cons of logistic regression include its assumption of linear decision boundaries, sensitivity to outliers, and potential limitations when dealing with non-linear relationships in the data.

**Naive Bayes**

Naive Bayes is a probabilistic machine learning algorithm used for classification tasks, particularly in situations where the features are independent of each other. It's based on Bayes' theorem, which describes the probability of a hypothesis given the evidence.

In Naive Bayes, the algorithm assumes that the features are conditionally independent given the class label, which means that the presence of one feature does not affect the presence of another. This simplifying assumption allows Naive Bayes to be computationally efficient and often leads to good performance, especially with small datasets.

Naive Bayes calculates the probability of each class given the input features using Bayes' theorem and then selects the class with the highest probability as the predicted class for the input.The algorithm requires estimating two types of probabilities:
1. Class Prior Probability: The probability of each class occurring in the dataset. It's calculated as the proportion of instances belonging to each class in the training data.
2. Class-Conditional Feature Probabilities: The probability of each feature value occurring given the class label. For continuous features, it's often modeled using probability density functions (e.g., Gaussian distribution), while for categorical features, it's calculated as the proportion of instances with each feature value within each class.Naive Bayes is known for its simplicity, speed, and ability to handle high-dimensional data well. It's commonly used in text classification tasks, such as spam detection and document categorization, where the "bag of words" assumption fits naturally.Despite its simplicity, Naive Bayes can perform surprisingly well in practice, especially when the independence assumption approximately holds or when there's limited training data available. However, this assumption may not always be valid in real-world datasets, and Naive Bayes may not perform as well when features are highly correlated.Overall, Naive Bayes is a powerful and easy-to-implement algorithm suitable for various classification tasks, particularly in situations where computational resources are limited or when dealing with high-dimensional data.

**Support Vector Machines (SVM)**

Support Vector Machines (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It's particularly effective for classification tasks in high-dimensional spaces.SVM aims to find the optimal hyperplane that best separates the data points into different classes. The hyperplane is defined as the decision boundary that maximizes the margin, which is the distance between the hyperplane and the nearest data points (called support vectors) from each class.In classification, SVM seeks to find the hyperplane that maximizes the margin while still correctly classifying all training data points. If the data is not linearly separable, SVM can use a kernel trick to map the input features into a higher-dimensional space where a hyperplane can separate the classes.SVM is effective in dealing with high-dimensional data and can handle cases where the number of features exceeds the number of samples. It's also robust against overfitting, especially in high-dimensional spaces.SVM uses a loss function called hinge loss, which penalizes misclassified points and encourages maximizing the margin. The optimization problem in SVM involves minimizing the hinge loss while also minimizing the norm of the weight vector (to promote a simpler model).Despite its effectiveness, SVMs can be sensitive to the choice of hyperparameters, such as the choice of kernel and regularization parameter. Additionally, SVMs can be computationally expensive, especially for large datasets.SVM has various applications in classification tasks, including text classification, image recognition, bioinformatics, and more. It's widely used in both academic research and industrial applications due to its versatility and performance.

**Model Evaluation:**

Assess model performance using metrics like accuracy, precision, recall, and F1-score.
Use the testing subset to evaluate generalization.
**Model architecture**

![spam](https://github.com/Shashankabasani/SPAM-SMS-DETECTION/assets/137595497/278553f9-b262-45f1-b5ae-edb5535ceed3)
Steps followed during model implementation 
Importing the raw data 
Extracting the features from the data
Describing the data set 
Perform feature engineering
 Splitting data 
Extracting the score from methods

**EXPERIMENTAL WORK** 
   
The experimental work undertaken for the research is as follows
Setting up an infrastructure  
google colab notebook
importing necessary libraries like NumPy, pandas, matplotlib, warnings, seaborn, and  sklearn 
reading the data using the Pandas library
defining data set and describing about it
making different types of graphs like heat map and bar graph using matplotlib and seaborn libraries
descriptive statistics 
scaling data into testing set and training set
finding the accuracy for the data using knn, random forest, and XGBoost simply by importing the particular libraries from sklearn library.
**Tools and Technologies:**
Developed using various tools and technologies, including:
Python programming language
Libraries like NumPy, Matplotlib, pandas and sklearn
Linear Regression, Random Forest, K-Nearest Neighbors (KNN), and XG BOOST models for predicting the score


**RESULTS**

The pie chart given below shows ham vs spam sms where 87.37% that is 4516 messages are ham sms and remaining 12.63% that is 653 sms are spam sms 

![image](https://github.com/Shashankabasani/SPAM-SMS-DETECTION/assets/137595497/e8da291d-bffb-4017-8f69-80b109b7ef43)

![newplot](https://github.com/Shashankabasani/SPAM-SMS-DETECTION/assets/137595497/5867b806-5475-484e-9a61-28a3dddd6410)

![newplot (1)](https://github.com/Shashankabasani/SPAM-SMS-DETECTION/assets/137595497/687937e5-8cb7-4ad1-9f3c-3b1f9146248e)

![newplot (2)](https://github.com/Shashankabasani/SPAM-SMS-DETECTION/assets/137595497/eb667709-48e8-4f93-b264-e936e4ac1b07)

**CONCLUSION AND FUTURE SCOPE**


We have already discussed all the important features of the datasets and their visualization in the above sections. But in order to conclude our report we would choose Logistic Regression . As we are satisfied by seeing how closely we have predicted the spam sms whith an accuracy of 95%.
There are still many technical indicators and feature variables that we have not included in our project, maybe there are some other indicators that we haven’t explored that would perform better. 
There are lots of Machine Learning algorithms that we haven’t tried and maybe a neural network or gradient boost would perform better than our solution.
Last but not least we have used data from the Canadian environment if we increase the data, we think the performance of our solution models may be improved.

