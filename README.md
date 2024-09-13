# NLP Resume Screening Project Report | Group 5

## Project Overview

In today's highly competitive job market, it is essential for recruitment processes to be efficient and effective. One of the most significant challenges faced by recruiters is the screening of resumes to find the right candidates. This project focuses on building a text classification model to automatically categorize resumes into predefined job categories using Natural Language Processing (NLP) techniques.

The main objective of this project is to classify resumes into one of 25 different job categories using machine learning algorithms. This will help recruiters quickly filter out relevant resumes, reducing manual efforts and ensuring that only the most suitable candidates are shortlisted.

---

## Methodology

### Data Collection

The dataset I used in this project consists of 962 resumes and their corresponding categories. Each resume is categorized into one of 25 job categories, including Java Developer, Data Science, Python Developer, DevOps Engineer, and others. The dataset is loaded into a pandas DataFrame and cleaned to ensure consistency.

**Dataset Overview:**
- Number of resumes: 962
- Number of categories: 25

### Data Preprocessing

To ensure that the data is ready for machine learning models, I performed the following preprocessing steps:
1. **Text Cleaning:** I implemented a function that converted all text to lowercase, removed special characters, numbers, URLs, and unnecessary spaces.
2. **Label Encoding:** I converted the categorical job categories into numerical labels for easier processing by machine learning algorithms.

### Exploratory Data Analysis (EDA)

I visualized the distribution of resumes across different job categories. This helped me understand the dataset better and identify any imbalances or patterns in the data.

**Resume Distribution:**
- The majority of resumes belong to categories like Java Developer, Testing, and DevOps Engineer, with each category having around 40 resumes on average.
- The graph below shows the distribution of resumes per category:

![Resume Distribution](insert_graph_here)

### Model Selection and Training

I used multiple machine learning algorithms to classify resumes into categories. The models included:
1. **Logistic Regression:** A basic linear model that is commonly used for text classification tasks.
2. **Linear Support Vector Classifier (LinearSVC):** A powerful linear classifier optimized for text data.
3. **K-Nearest Neighbors (KNN):** A distance-based classifier that predicts the category of a resume based on its neighbors in the feature space.
4. **Naive Bayes:** A probabilistic classifier well-suited for high-dimensional data like text.

#### Text Vectorization

I used **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to convert textual data into numerical features. This technique emphasizes important words in each document while down-weighting frequently occurring words across all documents.

### Hyperparameter Tuning

I performed hyperparameter tuning using **GridSearchCV** to optimize the performance of each model. The following parameters were tuned:
- **TF-IDF max features**: Number of features to be extracted.
- **N-gram range**: Uni-grams or bi-grams.
- **Classifier-specific parameters**: Regularization parameters for Logistic Regression and LinearSVC, and the number of neighbors for KNN.

---

## Results

### Model Evaluation

I evaluated each model based on precision, recall, F1-score, and overall accuracy. The models achieved high accuracy across all categories, with slight differences in performance. Here are the key results:

#### Logistic Regression
- **Accuracy:** 99%
- **Best Parameters:** `{'classifier__C': 10, 'classifier__penalty': 'l2', 'tfidf__max_features': 1000, 'tfidf__ngram_range': (1, 2)}`
- The model performed exceptionally well with high precision and recall for most categories.

#### LinearSVC
- **Accuracy:** 99%
- **Best Parameters:** `{'classifier__C': 0.1, 'tfidf__max_features': 3000, 'tfidf__ngram_range': (1, 1)}`
- This model provided robust results across all categories, proving to be a strong classifier for text data.

#### KNN
- **Accuracy:** 98%
- **Best Parameters:** `{'classifier__n_neighbors': 3, 'classifier__weights': 'distance', 'tfidf__max_features': 1000, 'tfidf__ngram_range': (1, 2)}`
- This model showed slightly lower performance compared to linear models but still achieved a high classification accuracy.

#### Naive Bayes
- **Accuracy:** 99%
- **Best Parameters:** `{'classifier__alpha': 0.1, 'tfidf__max_features': 1000, 'tfidf__ngram_range': (1, 1)}`
- The Naive Bayes model performed remarkably well and was computationally efficient.

### Business Insights

Automating the resume screening process using text classification techniques provides several business benefits:

1. **Efficiency:** Automating the classification of resumes saves considerable time and resources for recruiters. By using machine learning models, recruiters can quickly filter out resumes that don't match the job description, allowing them to focus on the most relevant candidates.
   
2. **Scalability:** This solution can easily scale to process thousands of resumes, making it suitable for large organizations that receive high volumes of applications.

3. **Consistency:** Manual resume screening can be subject to bias or human error. By leveraging machine learning, organizations can ensure consistency and objectivity in the initial stages of the hiring process.

4. **Cost Reduction:** The time saved by automating resume screening translates to lower operational costs. Recruitment teams can allocate their time to more critical tasks such as interviewing and candidate evaluation.

---

## Conclusion

This project successfully demonstrates how NLP and machine learning techniques can be used to automate the classification of resumes. With 99% accuracy across multiple models, this solution has the potential to revolutionize the recruitment process by making it faster, more efficient, and scalable. Future improvements can include extending the model to support multilingual resumes and incorporating more complex features to further improve classification accuracy.

---

## Future Work

- **Multi-language support:** I plan to implement models that can classify resumes written in multiple languages.
- **Advanced Feature Engineering:** I will incorporate semantic features using word embeddings like Word2Vec or transformers (e.g., BERT) to capture contextual meaning.
- **Model Deployment:** I plan to deploy the trained models into a production environment for real-time resume classification.

---

## References
- Libraries: Scikit-learn, Pandas, Numpy, Seaborn
