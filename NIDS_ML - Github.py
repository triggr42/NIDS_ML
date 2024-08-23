import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import time
import optuna
import hashlib


#File Paths
train_file_path = ''
test_file_path = ''

#Checksum and hash functions
def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def compute_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

#Compute annd verify file hashes
train_md5 = compute_md5(train_file_path)
test_md5 = compute_md5(test_file_path)
print(f"Train data MD5: {train_md5}")
print(f"Test data MD5: {test_md5}")

train_sha256 = compute_sha256(train_file_path)
test_sha256 = compute_sha256(test_file_path)
print(f"Train data SHA-256 {train_sha256}")
print(f"Test data SHA-256 {test_sha256}")

# Load data
train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

# Data exploration
print(train.head())
print(train.info())
print(train.describe())
print(train.describe(include='object'))
print(train.shape)
print(train.isnull().sum())

# Check for missing values and duplicates
total = train.shape[0]
missing_columns = [col for col in train.columns if train[col].isnull().sum() > 0]
for col in missing_columns:
    null_count = train[col].isnull().sum()
    per = (null_count / total) * 100
    print(f"{col}: {null_count} ({round(per, 3)}%)")

print(f"Number of duplicate rows: {train.duplicated().sum()}")

#Fill missing values with mean
#train.fillna(train.mean(), inplace=True)
#test.fillna(test.mean(), inplace=True)

#Drop duplicate values
#train = train.drop_duplicates()
#test = test.drop_duplicates()

# Visualize class distribution
sns.countplot(x=train['class'])
plt.show()

print('Class distribution Training set:')
print(train['class'].value_counts())

# Label Encoding
def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])

le(train)
le(test)

# Drop unnecessary column
train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)

# Feature selection
X_train = train.drop(['class'], axis=1)
Y_train = train['class']

rfc = RandomForestClassifier()
rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
selected_features = [v for i, v in feature_map if i]
print("Selected features:", selected_features)

X_train = X_train[selected_features]

# Scaling
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
test = scale.fit_transform(test)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)
print(f"x_train.shape: {x_train.shape}")
print(f"x_test.shape: {x_test.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"y_test.shape: {y_test.shape}")

# Logistic Regression
clfl = LogisticRegression(max_iter=1200000)
start_time = time.time()
clfl.fit(x_train, y_train.values.ravel())
end_time = time.time()
print("Training time: ", end_time - start_time)

#Logistic Regression Confusion Matrix and Classification Report
y_train_pred_lr = clfl.predict(x_train)
y_test_pred_lr = clfl.predict(x_test)

# Confusion Matrix and Classification Report for Logistic Regression
print("Logistic Regression - Training Set")
print(confusion_matrix(y_train, y_train_pred_lr))
print(classification_report(y_train, y_train_pred_lr))

print("Logistic Regression - Test Set")
print(confusion_matrix(y_test, y_test_pred_lr))
print(classification_report(y_test, y_test_pred_lr))

start_time = time.time()
y_test_pred = clfl.predict(x_test)
end_time = time.time()
print("Testing time: ", end_time - start_time)

lg_model = LogisticRegression(random_state=42)
lg_model.fit(x_train, y_train)
lg_train, lg_test = lg_model.score(x_train, y_train), lg_model.score(x_test, y_test)
print(f"Training Score: {lg_train}")
print(f"Test Score: {lg_test}")

# Optuna for hyperparameter tuning
def objective(trial):
    n_neighbors = trial.suggest_int('KNN_n_neighbors', 2, 16, log=False)
    classifier_obj = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier_obj.fit(x_train, y_train)
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy

study_KNN = optuna.create_study(direction='maximize')
study_KNN.optimize(objective, n_trials=1)
print(study_KNN.best_trial)

KNN_model = KNeighborsClassifier(n_neighbors=study_KNN.best_trial.params['KNN_n_neighbors'])
KNN_model.fit(x_train, y_train)
KNN_train, KNN_test = KNN_model.score(x_train, y_train), KNN_model.score(x_test, y_test)
print(f"Train Score: {KNN_train}")
print(f"Test Score: {KNN_test}")

y_train_pred_knn = KNN_model.predict(x_train)
y_test_pred_knn = KNN_model.predict(x_test)

# Confusion Matrix and Classification Report for KNN
print("K-Nearest Neighbors - Training Set")
print(confusion_matrix(y_train, y_train_pred_knn))
print(classification_report(y_train, y_train_pred_knn))

print("K-Nearest Neighbors - Test Set")
print(confusion_matrix(y_test, y_test_pred_knn))
print(classification_report(y_test, y_test_pred_knn))

# Decision Tree
clfd = DecisionTreeClassifier(criterion="entropy", max_depth=4)
start_time = time.time()
clfd.fit(x_train, y_train.values.ravel())
end_time = time.time()
print("Training time: ", end_time - start_time)

start_time = time.time()
y_test_pred = clfd.predict(x_train)
end_time = time.time()
print("Testing time: ", end_time - start_time)
def objective2(trial):
    dt_max_depth = trial.suggest_int('dt_max_depth', 2, 32, log=False)
    dt_max_features = trial.suggest_int('dt_max_features', 2, 10, log=False)
    classifier_obj = DecisionTreeClassifier(max_features=dt_max_features, max_depth=dt_max_depth)
    classifier_obj.fit(x_train, y_train)
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy

study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(objective2, n_trials=30)
print(study_dt.best_trial)

y_train_pred_dt = clfd.predict(x_train)
y_test_pred_dt = clfd.predict(x_test)

# Confusion Matrix and Classification Report for Decision Tree
print("Decision Tree - Training Set")
print(confusion_matrix(y_train, y_train_pred_dt))
print(classification_report(y_train, y_train_pred_dt))

print("Decision Tree - Test Set")
print(confusion_matrix(y_test, y_test_pred_dt))
print(classification_report(y_test, y_test_pred_dt))