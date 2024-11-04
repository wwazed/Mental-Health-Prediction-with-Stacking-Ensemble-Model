# **Mental Health Prediction with Stacking Ensemble Model**
This notebook demonstrates a machine learning pipeline for predicting mental health conditions based on demographic, lifestyle, and professional data. The project uses data from training and test sets, performs data cleaning and preprocessing, and leverages models for classification.
## **Dataset Description**
Both the training and testing datasets contained in this notebook include demographic, academic, and lifestyle-related attributes for individuals, with the intention of providing insights that can aid in the prediction of mental health status. Here is a breakdown of the key columns and their significance:
1. ID: Unique identifier for each individual.
2. Name: Name of the individual (likely not essential for modeling).
3. Gender: Gender of the individual, which may contribute to differences in mental health risk factors.
4. Age: Age of the individual, potentially important as mental health risks can vary across age groups.
5. City: The city where the individual resides, which might relate to urban vs. rural living conditions or regional stressors.
6. Working Professional or Student: Indicates if the individual is employed or a student, capturing different sources of pressure (academic vs. work-related).
7. Profession: Specific job role, which could provide additional context on work-related stress.
8. Academic Pressure: Numeric rating of academic pressure, applicable for students and possibly affecting stress levels.
9. Work Pressure: Numeric rating of job pressure, relevant to working professionals and can impact mental health.
10. CGPA: Cumulative Grade Point Average, representing academic performance and potentially related to academic satisfaction or pressure.
11. Study Satisfaction: Self-reported satisfaction with studies, which might influence overall well-being.
12. Job Satisfaction: Self-reported job satisfaction, potentially impacting mental health for working individuals.
13. Sleep Duration: Reported sleep duration category, which can correlate with mental health outcomes.
14. Dietary Habits: Self-reported diet quality (e.g., healthy, unhealthy), as diet can influence mood and stress levels.
15. Degree: Highest degree attained, potentially related to job satisfaction or career stress.
16. Suicidal Thoughts: Binary response to whether the individual has ever had suicidal thoughts, serving as a significant mental health indicator.
17. Work/Study Hours: Number of hours dedicated to work or study, which might contribute to burnout or stress.
18. Financial Stress: Numeric rating of financial stress, another factor often linked with mental health.
19. Family History of Mental Illness: Indicates whether there is a known family history of mental illness, a potential genetic or environmental risk factor.
20. Depression: The target variable, indicating whether the individual has been diagnosed with depression (binary).
The dataset’s 20 attributes are pivotal in constructing a robust mental health prediction model, as they provide a comprehensive understanding of the factors that influence mental well-being. Attributes such as age, gender, job satisfaction, academic pressure, financial stress, and family history of mental illness offer insights into personal, professional, and genetic factors that can impact mental health. This comprehensive range of features enables the model to discern intricate patterns across diverse lifestyles, thereby facilitating a more accurate and personalized approach to predicting mental health outcomes.

## **Data Preprocessing**
### **Missing Data Handling**
In the preprocessing stage of the dataset, handling missing values is a crucial step to ensure data quality and model performance. First, we identify the columns that contain missing values. Subsequently, we categorized the **numerical columns** and **categorical columns** from the dataframe. For numerical columns, we discovered that the attributes **Academic Pressure, Work Pressure, CGPA, Study Satisfaction, and Financial Stress & Job Satisfaction** have null or missing values. Notably, Financial Stress only has four null values. Therefore, for Financial Stress, we replaced the null values with the **median** of the column. For the remaining four attributes, we replaced the missing values with **-1**.

Consequently, all missing values have been rectified without resorting to oversampling. Subsequently, a histogram was constructed to ascertain the distribution of numerical values within the range of 0 to 10. Notably, the distribution for the age variable exhibited a range spanning from 18 to 60.

### **Encoding**
Encoding entails transforming categorical attributes, such as gender or profession, into numerical representations through specialized techniques. This transformation is accomplished using OrdinalEncoder, which assigns integer values to categorical columns, rendering them suitable for incorporation into machine learning algorithms.

From the correlation matrix, we observed that key factors such as “Academic Pressure,” “Work Pressure,” and “Financial Stress” exhibited significant correlations with one another, suggesting a complex interplay among these stress-related factors. The target variable, “Depression,” exhibited some correlation with factors like “Academic Pressure” and “Work Pressure,” implying that these factors may serve as potential predictors of mental health outcomes. Additionally, the heatmap revealed regions of high correlation within clusters, which could indicate multicollinearity among certain variables. This visualization facilitated the identification of influential features in predicting depression and those potentially redundant due to strong inter-correlations.

<p align="center">
  <img alt="Correlation" src="https://github.com/user-attachments/assets/5f54ea81-c3ae-453a-b341-668b2657ac10">
</p>

## Stacking Ensemble Model Creation
The Stacking Ensemble model employed a combination of **Random Forest (RF), K-Nearest Neighbors (K-NN), Support Vector Machine (SVM), and Decision Tree (DT)** base models. Their predictions were subsequently fused by a meta-model, **Logistic Regression**. We meticulously defined the base models with optimal parameters. For RF, we utilized **n_estimators = 200**, **10 nearest neighbors** for K-NN, and predefined base values for DT. For the meta-model, we employed Logistic Regression with predefined base values. Notably, no learning rate was determined. In all instances, the **random_state** parameter was fixed at 42. This approach helps capture different aspects of the data patterns, as each model has unique strengths.

<p align="center">
<img width="929" alt="Screenshot 2024-11-04 at 7 54 01 PM" src="https://github.com/user-attachments/assets/06aba928-db8e-4242-9edd-b2f34421902b">
</p>

The stacking model’s strength lies in its ability to mitigate the weaknesses of individual models. For instance, if one model underperforms on certain data segments, the others may compensate, leading to a more balanced and accurate prediction. 

## Training the Model
For training the model, we did not fix any learning rate. We retained the base values and set cv = 10. Subsequently, we split the dataset into 80% for training and 20% for validation. The training process took approximately 45 minutes.

## Evaluation
The model’s accuracy is evaluated on the validation data. This metric quantifies the frequency with which the model accurately predicts the target variable (depression) within the validation set. Consequently, it serves as a straightforward measure of the overall model’s performance. We achieved an accuracy of 94%. The classification report provides a comprehensive summary of the model’s performance across two classes (0 and 1). It encompasses metrics for precision, recall, and F1-score. For Class 0 (22,986 instances), the model demonstrated high accuracy, achieving a precision of 0.95, recall of 0.97, and F1-score of 0.96. This indicates the model’s effectiveness in identifying this class. Conversely, Class 1 (5,154 instances) exhibited slightly lower performance, with precision of 0.85, recall of 0.79, and F1-score of 0.82. This suggests some misclassifications. Furthermore, the macro average F1-score was 0.89, while the weighted average F1-score was 0.94. These values reflect balanced and reliable performance, particularly for the larger class.
<p align="center">
  <img width="417" alt="Screenshot 2024-11-04 at 8 02 53 PM" src="https://github.com/user-attachments/assets/482e3be3-9d30-4104-8e39-bd2e230d1d70">
</p>

### **ROC Score**
The ROC curve plots the model's performance across different classification thresholds, illustrating the trade-off between FPR and TPR. The curve is close to the top left corner, indicating strong discriminatory ability. The Area Under the Curve (AUC) score is **0.97**, signifying excellent performance; an AUC of 1.0 represents a perfect model, while 0.5 indicates random guessing. This high AUC suggests that the Stacking Ensemble model effectively differentiates between the positive and negative classes, providing reliable predictions.
<p align="center">
  <img alt="ROC CURVE" src="https://github.com/user-attachments/assets/34dd26af-1ca0-4a72-8c17-7135f3ad88c1">
</p>

## **Required Libraries**
1. Python = 3.12.7
2. Pandas
3. NumPy
4. Seaborn
5. Scikit-learn

## **Contributions**
Contributions are welcome! Feel free to open issues or submit pull requests.

## **Licence**
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

**Note:** The dataset has been retrieved from Kaggle’s competition. Upon the conclusion of the competition, the dataset will become publicly available and can be shared via a link provided here.
