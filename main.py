# PACKAGES
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing, model_selection, pipeline
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import random
import matplotlib.pyplot as plt
import joblib
import requests

st.title("Prediction of the granting of a credit")

st.sidebar.header("Customer ID")
data_train2 = pd.read_csv('train2.csv')
data_train2 = data_train2.head(2000)
clients = random.choices(data_train2.SK_ID_CURR.tolist(),k = 50)
client_id = st.sidebar.selectbox("ID client", clients)
st.write(client_id)

# MODELISATION
X = data_train2.drop(["TARGET","SK_ID_CURR","TARGET_name"], axis = 1)
y = data_train2.TARGET

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regressor = RandomForestClassifier()
params = {
    "n_estimators":[50,100],
    "max_depth":[2,4,8]
}

gsv = model_selection.GridSearchCV(regressor, params, cv=5)
gsv.fit(X_train_scaled,y_train)


profil = data_train2[data_train2["SK_ID_CURR"] == int(client_id)].drop(["TARGET","SK_ID_CURR","TARGET_name"], axis = 1)
prediction = gsv.predict(profil)
prediction_proba = gsv.predict_proba(profil)

# METRICS
y_pred = gsv.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_train, gsv.predict_proba(X_train_scaled)[:, 1])

# PIPELINE
pipeline = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                              ('regressor', RandomForestClassifier(**gsv.best_params_))])
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
joblib.dump(pipeline, 'pipeline_credit.joblib')

# Déploiement mlflow
signature = infer_signature(X_train, y_train)
#mlflow.sklearn.save_model(pipeline, 'mlflow_model', signature=signature)


st.header("Metrics are :")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Accuracy")
    st.write(f'Accuracy =', acc)

with col2:
    st.subheader("Roc_auc_score")
    st.write(f'roc_auc_score =', roc)


st.header("Decision is :")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.subheader("Prediction")
    st.write(data_train2.TARGET[prediction])

with col2:
    st.subheader("Probability")
    st.write(prediction_proba)

with col3:
    st.subheader("Target name")
    st.write(data_train2.TARGET_name[prediction])

with col4:
    st.subheader("Gender")
    Gender = data_train2[data_train2["SK_ID_CURR"] == int(client_id)]["CODE_GENDER"].values[0]
    st.write(Gender)

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception("Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
client_data = data_train2[data_train2['SK_ID_CURR'] == client_id][["CODE_GENDER","CNT_CHILDREN",'AMT_INCOME_TOTAL','AMT_CREDIT',
              'REGION_POPULATION_RELATIVE','EXT_SOURCE_3','CREDIT_TERM',"YEARS_EMPLOYED",'YEARS_REGISTRATION',
              'YEARS_ID_PUBLISH']].values
predict_btn = st.button('Prédire')
if predict_btn:
     data = client_data
     pred = request_prediction(MLFLOW_URI, data)[0]
     st.write('Le crédit {:.2f}'.format(pred)) 


       
st.subheader("Graphics:")
# AMT_INCOME_TOTAL
fig = plt.figure(figsize=(6,4))
plt.title("Distribution of income for credit applicants")
plt.ylabel('Density', fontsize=12)
plt.xlabel("AMT_INCOME_TOTAL", fontsize=12)
plt.hist(data_train2["AMT_INCOME_TOTAL"],color='yellow')
income = data_train2[data_train2["SK_ID_CURR"] == int(client_id)]["AMT_INCOME_TOTAL"].values[0]
plt.axvline(income,color='k',linestyle='dashed')
st.pyplot(fig)

# CNT_CHILDREN
fig = plt.figure(figsize=(6,4))
plt.title("Distribution of the number of children for credit applicants")
plt.ylabel('Density', fontsize=12)
plt.xlabel('CNT_CHILDREN', fontsize=12)
plt.hist(data_train2["CNT_CHILDREN"], color='red')
children = data_train2[data_train2["SK_ID_CURR"] == int(client_id)]["CNT_CHILDREN"].values[0]
plt.axvline(children,color='k',linestyle='dashed')
st.pyplot(fig)

# Credit
fig = plt.figure(figsize=(6,4))
plt.title("Distribution of credit for credit applicants")
plt.ylabel('Density', fontsize=12)
plt.xlabel("AMT_CREDIT", fontsize=12)
plt.hist(data_train2["AMT_CREDIT"])
credit = data_train2[data_train2["SK_ID_CURR"] == int(client_id)]["AMT_CREDIT"].values[0]
plt.axvline(credit,color='k',linestyle='dashed')
st.pyplot(fig)

#CODE_GENDER
# Gender = data_train2[data_train2["SK_ID_CURR"] == int(client_id)]["CODE_GENDER"].values[0]
# st.sidebar.selectbox('Gender',('male','female'))

# YEARS_EMPLOYED
fig = plt.figure(figsize=(6,4))
plt.title("Distribution of years employed for credit applicants")
plt.ylabel('Density', fontsize=12)
plt.xlabel("YEARS_EMPLOYED", fontsize=12)
plt.hist(data_train2["YEARS_EMPLOYED"],color='green')
employed = data_train2[data_train2["SK_ID_CURR"] == int(client_id)]["YEARS_EMPLOYED"].values[0]
plt.axvline(employed,color='orange',linestyle='dashed')
st.pyplot(fig)