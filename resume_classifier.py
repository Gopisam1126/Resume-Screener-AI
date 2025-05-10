import pandas as pd
import re
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("resume_data.csv")
le = LabelEncoder()
tfidf = TfidfVectorizer(stop_words='english')

max_count = df['Category'].value_counts().max()

bal_df = (
    df
    .groupby('Category')
    .apply(lambda x: x.sample(max_count, replace=True), include_groups=False)
    .reset_index(drop=False)
)

df = bal_df.sample(frac=1).reset_index(drop=True)
df = df.drop(columns="level_1")

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))
df['Category'] = le.fit_transform(df['Category'])

X  = tfidf.fit_transform(df['Resume'])
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_params = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 300],
            "max_depth": [None, 10, 30],
            "min_samples_split": [2, 5],
        }
    },
    "SVC": {
        "model": SVC(random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs", "saga"],
        }
    },
}

best_estimators = {}
for name, mp in model_params.items():
    print(f"Running GridSearchCV for {name}...")
    grid = GridSearchCV(
        estimator=mp["model"],
        param_grid=mp["params"],
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
        verbose=1
    )
    grid.fit(X_train, y_train)
    best_estimators[name] = grid.best_estimator_
    print(f"  Best params for {name}: {grid.best_params_}")
    print(f"  Best CV accuracy: {grid.best_score_:.4f}\n")

for name, model in best_estimators.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"=== {name} ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\n")
    
svc_model = SVC(C=1, gamma='scale', kernel='linear')

svc_model.fit(X_train, y_train)

def pred(input_resume):
    cleaned_text = cleanResume(input_resume) 
    vec_txt = tfidf.transform([cleaned_text])
    vec_txt = vec_txt.toarray()

    predicted_category = svc_model.predict(vec_txt)

    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]

joblib.dump({
    'model': svc_model,
    'vectorizer': tfidf,
    'label_encoder': le
}, 're_sc_pipeline.pkl')