import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

header= st.container()
descpriton=st.container()
dataset= st.container()
visualisation=st.container()
Preprocessing = st.container()
feature_engineering=st.container()
model=st.container()
user_model= st.container()


with header :
    st.title("Welcome to my project!")
    
with descpriton:
    st.header("Loan Amount Prediction")
    st.write("This is a Binary Classification problem in which we need to predict our Target label which is “Loan Status”")
    st.write("Loan status can have two values: Yes or NO.")
    st.write("Yes: if the loan is approved")
    st.write("NO: if the loan is not approved")
    st.write("So using the training dataset we will train our model and try to predict our target column that is “Loan Status” on the test dataset.")

with dataset :
    #training set
    df=pd.read_csv("data/train_set.csv")
    df.drop("Loan_ID", axis=1, inplace=True)

    #test set
    tf = pd.read_csv("data/test_set.csv")
    tf.drop("Loan_ID", axis=1, inplace=True)

    st.write("This is our raw training dataset")
    st.write(df)
    st.write("Shape of the dataset", df.shape)
    
    st.write("This is our raw test set")
    st.write(tf)
    st.write("Shape of the dataset", tf.shape)


with visualisation:
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.header("Data visulaisation")

    st.write("The given dataset can be splitted into categorical and numerical columns.")
    st.write("* **Categorical columns** =['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']")
    st.write("* **Numerical columns** =['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']")

    st.subheader("Categorical columns")
    categorical_columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']
    fig, axes = plt.subplots(4, 2, figsize=(12, 15))
    for idx, cat_col in enumerate(categorical_columns):
        row, col = idx // 2, idx % 2
        sns.countplot(x=cat_col, data=df, hue='Loan_Status', ax=axes[row, col])
    st.pyplot()

    st.write("Plots above convey following things about the dataset:")
    st.write("1) **Loan Approval Status**: About 2/3rd of applicants have been granted loan.")
    st.write("2) **Sex**: There are more Men than Women (approx. 3x)")
    st.write("3) **Martial Status**: 2/3rd of the population in the dataset is Marred; Married applicants are more likely to be granted loans.")
    st.write("4) **Dependents**: Majority of the population have zero dependents and are also likely to accepted for loan.")
    st.write("5) **Education**: About 5/6th of the population is Graduate and graduates have higher propotion of loan approval.")
    st.write("6) **Employment**: 5/6th of population is not self employed.")
    st.write("7) **Property Area**: More applicants from Semi-urban and also likely to be granted loans.")
    st.write("8) Applicant with **credit history** are far more likely to be accepted.")
    st.write("9) **Loan Amount Term**: Majority of the loans taken are for 360 Months (30 years).")

    #Numerical columns
    st.subheader("Numerical columns")
    plt.figure(figsize=(6, 3))
    sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df)
    st.pyplot()
    plt.figure(figsize=(6, 3))
    sns.boxplot(x='Loan_Status', y='CoapplicantIncome', data=df)
    st.pyplot()
    plt.figure(figsize=(6, 3))
    sns.boxplot(x='Loan_Status', y='LoanAmount', data=df)
    st.pyplot()

    st.write(""" ##### Feature transformation""")
    st.write("All the numerical features/columns posses right skew as we know if our feature follow bell curve of distribution then it is easier to produce a good model so we will do feature transformation.")
    st.write("For **outlier** treatment of Total_Amount and LoanAmount we use the log transformation method. Due to these outliers, the bulk of the data is at the left with a long right tail. This is called right skewness and **log transformation** is a good means to remove outliers. The log transformation doesn’t affect smaller instances rather reduces larger values to get a normal distribution.")

with Preprocessing:
    # training set
    def EDA():
        for i in [df]:
            i["Gender"] = i["Gender"].fillna(df.Gender.dropna().mode()[0])
            i["Married"] = i["Married"].fillna(df.Married.dropna().mode()[0])
            i["Self_Employed"] = i["Self_Employed"].fillna(df.Self_Employed.dropna().mode()[0])
            i["Dependents"] = i["Dependents"].fillna(df.Dependents.dropna().mode()[0])
            i["Credit_History"] = i["Credit_History"].fillna(df.Credit_History.dropna().mode()[0])

        df1 = df.loc[:, ["LoanAmount", "Loan_Amount_Term"]]
        imp = IterativeImputer(RandomForestRegressor(), max_iter=100, random_state=0)
        df1 = pd.DataFrame(imp.fit_transform(df1), columns=df1.columns)

        df["LoanAmount"] = df1["LoanAmount"]
        df["Loan_Amount_Term"] = df1["Loan_Amount_Term"]

        col = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Loan_Status"]
        l = LabelEncoder()
        for i in col:
            df[i] = l.fit_transform(df[i])
        return df

    df = EDA()

    # test set
    def EDA_2():
        for i in [tf]:
            i["Gender"] = i["Gender"].fillna(tf.Gender.dropna().mode()[0])
            i["Married"] = i["Married"].fillna(tf.Married.dropna().mode()[0])
            i["Self_Employed"] = i["Self_Employed"].fillna(tf.Self_Employed.dropna().mode()[0])
            i["Dependents"] = i["Dependents"].fillna(tf.Dependents.dropna().mode()[0])
            i["Credit_History"] = i["Credit_History"].fillna(tf.Credit_History.dropna().mode()[0])

        tf1 = tf.loc[:, ["LoanAmount", "Loan_Amount_Term"]]
        imp = IterativeImputer(RandomForestRegressor(), max_iter=100, random_state=0)
        tf1 = pd.DataFrame(imp.fit_transform(tf1), columns=tf1.columns)

        tf["LoanAmount"] = tf1["LoanAmount"]
        tf["Loan_Amount_Term"] = tf1["Loan_Amount_Term"]

        col = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
        l = LabelEncoder()
        for i in col:
            tf[i] = l.fit_transform(tf[i])
        return tf

    tf = EDA_2()

    st.write(""" ##### Correlation plot""")
    plt.figure(figsize=(15, 9))
    sns.heatmap(df.corr(), cmap='cubehelix_r', annot=True)
    st.pyplot()


with feature_engineering:
    #Feature Engineering

    # Creation of new variable
    df["Total_income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    tf["Total_income"] = tf["ApplicantIncome"] + tf["CoapplicantIncome"]

    # applying log transformation to the attribute to transform into bell shape curve from right skew

    # Training set
    df["ApplicantIncome_log"] = np.log(df["ApplicantIncome"])
    df["CoapplicantIncome_log"] = np.log(df["CoapplicantIncome"])
    df["LoanAmount_log"] = np.log(df["LoanAmount"])
    df["Total_income_log"] = np.log(df["Total_income"])

    # Test set
    tf["ApplicantIncome_log"] = np.log(tf["ApplicantIncome"])
    tf["CoapplicantIncome_log"] = np.log(tf["CoapplicantIncome"])
    tf["LoanAmount_log"] = np.log(tf["LoanAmount"])
    tf["Total_income_log"] = np.log(tf["Total_income"])

    # dropping of columns with high co-relation with itself
    df.drop(columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_income','ApplicantIncome_log', 'CoapplicantIncome_log'], axis=1, inplace=True)
    tf.drop(columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_income','ApplicantIncome_log', 'CoapplicantIncome_log'], axis=1, inplace=True)

    st.subheader('Final datset')
    if st.checkbox('Show modified dataset after prepocessing'):
        st.write("This is our final data set after preprocessing and EDA.")
        st.write(df)
    # st.write(df)

def add_parameter_ui(clf_name):
    params=dict()
    if clf_name=="KNN":
        K=st.sidebar.slider("The value of K",1,15)
        params["K"]=K
    elif clf_name=="SVM":
        C=st.sidebar.slider("The value of C",0.01,10.0)
        params["C"]=C
    elif clf_name=="Decision Tree":
        max_depth_D = st.sidebar.slider("max_depth", 2, 20)
        params["max_depth_D"] = max_depth_D
    else :
        max_depth_R=st.sidebar.slider("max_depth",5,50)
        n_estimators=st.sidebar.slider("n_estimators",10,200,step=10)
        params["max_depth_R"]=max_depth_R
        params["n_estimators"] = n_estimators
    return params

def get_classifier(clf_name,params):

    if clf_name=="KNN":
       clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name=="SVM":
       clf = SVC(C=params["C"])
    elif clf_name=="Decision Tree":
        clf=DecisionTreeClassifier(max_depth=params["max_depth_D"],random_state=42)
    else :
       clf= RandomForestClassifier(max_depth=params["max_depth_R"],n_estimators=params["n_estimators"],random_state=42)
    return clf

with model:

    y = df["Loan_Status"]
    Df = df
    Df.drop("Loan_Status", axis=1, inplace=True)
    X = Df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    @st.cache(suppress_st_warning=True)
    def randomforest():
        # Defining the instance of RandomForest
        rfc = RandomForestClassifier(random_state=1024)

        # Defining the parameter for using GridSearchCV()
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 250],  # no of decision tree
            'max_depth': [4, 5, 6, 7, 8],  # depth of tree
        }

        # We are using GridSearchCV for best hyperparameter tuning
        GS_CV = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        GS_CV.fit(X_train, y_train)

        st.header("Random Forest Classifier")
        st.write("After applying GridSearchCV for best hyperparameter tuning we get these parameters\n.")
        st.write(""" #### BEST PARAMETERS """)

        # Final selection of parameter model
        A = GS_CV.best_params_
        st.write(A)

        Random_forest = RandomForestClassifier(random_state=42, n_estimators=A['n_estimators'],
                                               max_depth=A['max_depth'])
        Random_forest.fit(X_train, y_train)
        y_predict = Random_forest.predict(X_test)
        st.write(""" ##### CONFUSION MATRIX""")
        st.write(confusion_matrix(y_test, y_predict))

        st.write(""" ##### MODEL EVALUATION""")
        st.write(f"Accuracy of this  model: {accuracy_score(y_test, y_predict)}")
        st.write(f" F1 score of this model: {f1_score(y_test, y_predict)}")
        st.write("Validation Mean Accuracy: ",
                 cross_val_score(Random_forest, X_test, y_test, cv=5, scoring='accuracy').mean())
        st.write("Validation Mean F1 Score: ",
                 cross_val_score(Random_forest, X_test, y_test, cv=5, scoring='f1_macro').mean())

        y_pred2 = Random_forest.predict(tf)
        b = pd.DataFrame(y_pred2)
        b.replace({1: "Y", 0: "N"}, inplace=True)
        b.rename(columns={b.columns[0]: "Predicted Values"})
        st.subheader("Predicted Loan_Status using the Random Forest classifier")
        st.write(b)
        colors = ['green']
        plt.figure(figsize=(10, 6))
        plt.hist(b, color=colors)
        plt.title("Histogram plot", fontweight="bold")
        plt.xlabel("Loan_Status")
        plt.ylabel("Count")
        st.pyplot()

    randomforest()


with user_model:

    st.sidebar.title("Try on your own classifiers")
    classifier_name = st.sidebar.selectbox("Select your algorithm", ("KNN", "SVM","Decision Tree", "Random Forest"))
    params = add_parameter_ui(classifier_name)
    clf = get_classifier(classifier_name, params)

    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)

    st.header(f"Model training using the chosen classifier - {classifier_name}")
    st.write(f"Accuracy of your model: {accuracy_score(y_test,y_pred)}")
    st.write(f" F1 score of your model: {f1_score(y_test, y_pred)}")
    st.write("Validation Mean Accuracy: ", cross_val_score(clf, X_test, y_test, cv=5, scoring='accuracy').mean())
    st.write("Validation Mean F1 Score: ", cross_val_score(clf, X_test, y_test, cv=5, scoring='f1_macro').mean())

    y_pred1 = clf.predict(tf)
    a = pd.DataFrame(y_pred1)
    a.replace({1: "Y", 0: "N"}, inplace=True)
    a.rename(columns={a.columns[0]: "Predicted Values"})
    st.subheader("Predicted Loan_Status using the chosen classifier")
    st.write(a)












