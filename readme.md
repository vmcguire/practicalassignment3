# Customer Subscription Prediction - Practical Application III
### Victor McGuire

## Overview:
In this project, we aimed to predict customer subscriptions for a term deposit using different classifiers: K Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines. 

The dataset contains various features related to bank client data, last contact of the current campaign, and social and economic context attributes. 

Our goal was to compare the performance of these classifiers and identify the most effective model for predicting customer subscriptions. We also conducted hyperparameter tuning for each model to achieve optimal results.

## Data:
We worked with a dataset containing bank client information, campaign-related variables, and social-economic indicators. The dataset had a total of 41,188 rows and 21 columns, with the target variable being whether the client subscribed to a term deposit or not (binary: 'yes' or 'no').

## Approach:
1. Data Exploration: We started by exploring the dataset to understand the distribution of features, identify missing values, and gain insights into the target variable's distribution.
2. Data Preprocessing: We performed data preprocessing tasks, including handling categorical variables through one-hot encoding and ensuring data consistency and integrity.
3. Model Selection: We selected four classifiers for comparison: K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.
4. Train-Test Split: We split the data into training and testing sets to evaluate each model's performance on unseen data.
5. Hyperparameter Tuning: We used grid search with cross-validation to find the best hyperparameters for each model, including C, multi_class, penalty, solver, criterion, max_depth, min_samples_leaf, min_samples_split, kernel, and gamma.
6. Model Training and Evaluation: Each classifier was trained on the training set, and their accuracy was evaluated on the test set.
7. Model Comparison: We compared the performance of the classifiers based on their accuracy scores.

## Findings:
After evaluating the classifiers on the original 7 columns of data, we obtained the following accuracy scores:

* Logistic Regression: Accuracy = 0.8876 (C: 0.1, multi_class: auto, penalty: l2, solver: newton-cg, verbose: 0)
* Decision Tree: Accuracy = 0.8869 (criterion: entropy, max_depth: 5, min_samples_leaf: 1, min_samples_split: 2)
* Support Vector Machines (SVM): Accuracy = 0.8864 (C=1, kernel = rbf, gamma = auto)
* K Nearest Neighbors (KNN): Accuracy = 0.8728 (n_neighbors = 105, weights = distance)

After obtaining these initial accuracy scores, we decided to explore a new dataset that included additional features. We applied the same Logistic Regression model with the selected parameters (C: 0.1, multi_class: auto, penalty: l2, solver: newton-cg, verbose: 0) on this new dataset. Surprisingly, the accuracy of the Logistic Regression model improved significantly to 0.8973 on this new dataset, outperforming the other classifiers and its own performance on the original 7 columns.

The inclusion of additional features appears to have enhanced the model's predictive power, resulting in better accuracy in predicting customer subscriptions for term deposits. It is crucial to acknowledge the importance of feature engineering and exploring relevant attributes to improve model performance.

Therefore, we recommend using the Logistic Regression model with the selected parameters (C: 0.1, multi_class: auto, penalty: l2, solver: newton-cg, verbose: 0) on the extended dataset for predicting customer subscriptions, as it demonstrated the highest accuracy among all the classifiers we tested. Nonetheless, continuous monitoring and evaluation of the model's performance are essential to ensure its effectiveness over time.

## Recommendations:
Based on our findings, the Logistic Regression model with hyperparameters (C: 0.1, multi_class: auto, penalty: l2, solver: newton-cg, verbose: 0) performed the best with an accuracy of 0.8973. We recommend using this model for predicting customer subscriptions for term deposits as it outperformed the other classifiers. However, it's important to continuously monitor and update the model as new data becomes available to ensure its predictive power remains optimal.

## Notebook Link:
For a more detailed view of the project, including data exploration, preprocessing, model training, evaluation, and hyperparameter tuning, please refer to the Jupyter Notebook available at [View prompt_III.ipynb](https://github.com/vmcguire/practicalassignment3/blob/main/prompt_III.ipynb).