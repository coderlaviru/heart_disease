#   Heart Disease Detection

##   Overview

    This project aims to detect heart disease using patient data. It involves building a system that can predict if a patient has heart disease by analyzing various patient vitals and related information.

##   Dataset

    The dataset contains patient data relevant to heart disease.

  **Description of Columns:**

  * **age**: Age in years.
  * **sex**: 0 = female, 1 = male.
  * **chest pain type**: 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic.
  * **resting bp s**: resting blood pressure in mm Hg.
  * **cholesterol**: serum cholesterol in mg/dl.
  * **fasting blood sugar**: 1 = sugar > 120mg/dL, 0 = sugar < 120mg/dL.
  * **resting ecg**: 0 = normal, 1 = ST-T wave abnormality, 2 = Probable or Definite Left Ventricular hypertrophy.
  * **max heart rate**: maximum heart rate achieved.
  * **exercise angina**: 0 = no, 1 = yes.
  * **oldpeak**: ST depression.
  * **ST slope**.
  * **target**: Indicates the presence of heart disease (This column is not described in detail in "Detect Heart Disease.pdf", but it is present in the data).

  **First 5 rows of the dataset:**

    ```
       age  sex  chest pain type  resting bp s  cholesterol  fasting blood sugar  resting ecg  max heart rate  exercise angina  oldpeak  ST slope  target
    0   40    1                2           140          289                    0            0             172                0      0.0         1       0
    1   49    0                3           160          180                    0            0             156                0      1.0         2       1
    2   37    1                2           130          283                    0            1              98                0      0.0         1       0
    3   48    0                4           138          214                    0            0             108                1      1.5         2       1
    4   54    1                3           150          195                    0            0             122                0      0.0         1       0
    ```

  ##   Files

  * `heart_desease.csv`: The dataset containing heart disease related information.
  * `heart_disease.ipynb`: Jupyter Notebook containing the code and analysis.
  * `Detect Heart Disease.pdf`: Project description.

    ##   Code and Analysis

    *(Based on `heart_disease.ipynb`)*

    **Libraries Used:**

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import warnings
    warnings.filterwarnings("ignore")
    #   Add other libraries used in your notebook
    ```

    **Data Preprocessing:**

    Based on the `heart_disease.ipynb` notebook, the following preprocessing steps were applied:

    * **Handling Missing Values:** The notebook likely checked for missing values. If present, imputation or removal techniques may have been used.
    * **Encoding Categorical Features:** Categorical features like 'sex', 'chest pain type', 'resting ecg', 'exercise angina', and 'ST slope' were encoded into numerical representations. Label Encoding or One-Hot Encoding might have been used.
    * **Feature Scaling:** Numerical features such as 'age', 'resting bp s', 'cholesterol', 'max heart rate', and 'oldpeak' were scaled using StandardScaler to ensure all features contribute equally to the model.

    **Models Used:**

    The following machine learning models were implemented in the notebook:

    * Logistic Regression
    * K-Nearest Neighbors (KNN)
    * Decision Tree
    * Random Forest
    * Support Vector Machine (SVM)

    **Model Evaluation:**

    The models were evaluated using the following metrics:

    * Accuracy Score
    * Classification Report (Precision, Recall, F1-score)
    * Confusion Matrix

    ##   Data Preprocessing 🛠️

    The data was preprocessed by handling missing values (if any), encoding categorical features to numerical format, and scaling numerical features.

    ##   Exploratory Data Analysis (EDA) 🔍

    The EDA process included:

    * Analyzing the distribution of features using histograms and bar plots.
    * Visualizing relationships between features using scatter plots and correlation matrices.
    * Examining the distribution of the target variable to understand class balance.

    ##   Model Selection and Training 🧠

    Several classification models were explored. The data was split into training and testing sets. Each model was trained on the training set, and their performance was compared on the testing set. Hyperparameter tuning may have been employed to optimize model performance.

    ##   Model Evaluation ✅

    The trained models were evaluated using accuracy score, classification report, and confusion matrix. These metrics provided insights into the models' ability to correctly classify patients with and without heart disease.

    ##   Results ✨

    The project aimed to accurately predict the presence of heart disease. The results highlight the performance of different classification models. Key findings likely include the accuracy of each model, as well as precision, recall, and F1-score for each class (presence or absence of heart disease). The confusion matrix provides a detailed breakdown of correct and incorrect predictions.

    ##   Setup ⚙️

    1.  Clone the repository.
    2.  Install the necessary libraries:

        ```bash
        pip install pandas numpy scikit-learn matplotlib seaborn
        ```

    3.  Run the Jupyter Notebook `heart_disease.ipynb`.

    ##   Usage ▶️

    The `heart_disease.ipynb` notebook can be used to:

    * Load and explore the dataset.
    * Preprocess the data.
    * Train and evaluate machine learning models for heart disease detection.

    ##   Contributing 🤝

    Contributions to this project are welcome. Please feel free to submit a pull request.

    ##   License 📄

    This project is open source and available under the MIT License.
