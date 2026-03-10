This project uses Machine Learning techniques to analyze the CarDekho dataset and build a predictive model. The goal is to preprocess the data, perform exploratory analysis, engineer useful features, and train a machine learning model to make predictions.

 Project Workflow:
 Importing Libraries

First, the required Python libraries are imported to perform data analysis and machine learning tasks.

Libraries used:

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn


Importing the Dataset:
The dataset is loaded using the Pandas library.
df = pd.read_csv("car data.csv")


 Understanding the Data:
To understand the dataset, we check:

Shape of the dataset

Data types

Summary statistics

Missing values


df.head()
df.info()
df.describe()

 Identifying Numerical and Categorical Features:

The dataset is divided into:

Numerical Features

Example:

Present Price

Kms Driven

Owner


Categorical Features

Example:

Fuel Type

Seller Type

Transmission


This helps in selecting appropriate preprocessing techniques.


---

5️⃣ Checking Duplicate Values

Duplicate rows in the dataset are checked and removed if necessary.

df.duplicated().sum()
df.drop_duplicates(inplace=True)


---

📊 Exploratory Data Analysis (EDA)

EDA is performed to understand relationships between variables using:

Bar plots

Scatter plots

Correlation heatmap

Distribution plots


Example:

sns.heatmap(df.corr(), annot=True)

This helps identify patterns and important features affecting the target variable.


---

⚙️ Feature Engineering

Feature engineering improves the model performance by modifying or creating useful features.

Tasks performed

1. Removing Unnecessary Columns

Columns that do not contribute to prediction are removed.

df.drop(['Car_Name'], axis=1, inplace=True)

2. Converting Categorical Variables to Numerical

Categorical columns are converted using One-Hot Encoding.

df = pd.get_dummies(df, drop_first=True)


---

🎯 Dividing X and Y Variables

The dataset is separated into:

X (Independent Variables / Features)

Y (Dependent Variable / Target)


X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']


---

🔀 Train-Test Split

The dataset is divided into training and testing sets to evaluate the model.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


---

📏 Feature Scaling

Feature scaling is applied to normalize the numerical values.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

 Model Building:
Machine Learning models are used to train and predict the car selling price.

Example models:

Linear Regression

Random Forest

Decision Tree


Example:

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


Model Evaluation:
Model performance is evaluated using metrics such as:

R² Score

Mean Absolute Error

Mean Squared Error


from sklearn.metrics import r2_score

pred = model.predict(X_test)
r2_score(y_test, pred)


 Tools & Technologies:

Python

Jupyter Notebook

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn


 Conclusion

This project demonstrates the complete Machine Learning workflow, including:

Data preprocessing

Exploratory Data Analysis

Feature Engineering

Model training and evaluation


The trained model can be used to predict the selling price of cars based on different features.
