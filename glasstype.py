import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Droping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)

    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
    glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "building windows float processed".upper()
    elif glass_type == 2:
        return "building windows non float processed".upper()
    elif glass_type == 3:
        return "vehicle windows float processed".upper()
    elif glass_type == 4:
        return "vehicle windows non float processed".upper()
    elif glass_type == 5:
        return "containers".upper()
    elif glass_type == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()

# Add title on the main page and in the sidebar.
st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis")

# Using if statement, display raw data on the click of the checkbox.
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Full Dataset")
    st.dataframe(glass_df)

st.sidebar.subheader("Scatter plot")
# Remove deprecation warning.
st.set_option("deprecation.showPyplotGlobalUse",False)
# Choosing x-axis values for scatter plots.
features_list = st.sidebar.multiselect("Select the x-axis values:", 
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Creating scatter plots.
for feature in features_list:
    st.subheader(f"Scatter plot between {feature} and GlassType")
    plt.figure(figsize = (12, 6))
    sns.scatterplot(x = feature, y = 'GlassType', data = glass_df)
    st.pyplot()
# Remove the code blocks for histogram and box plots.

# Add a subheader in the sidebar with label "Visualisation Selector"
st.sidebar.subheader("Visualisation Selector")
plot_types = st.sidebar.multiselect("Select the plots:", 
                                            ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))
# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
# and with 6 options passed as a tuple ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot').
# Store the current value of this widget in a variable 'plot_types'.
if 'Histogram' in plot_types:
  # plot histogram

  feature = st.sidebar.selectbox("Select the x-axis values:", 
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  
  st.subheader("Histogram")
  plt.figure(figsize = (12, 6))
  plt.hist(glass_df[feature], bins = 'sturges', edgecolor = 'black')
  st.pyplot()
if 'Box Plot' in plot_types:
  # plot box plot
  st.sidebar.subheader("Box Plot")

  # Choosing columns for box plots.

  # Create box plots.
  st.subheader(f"Box plot for {col}")
  plt.figure(figsize = (12, 2))
  sns.boxplot(glass_df[col])
  st.pyplot() 
if 'Count Plot' in plot_types:
  # plot count plot 
  st.subheader("Count Plot")
  plt.figure(figsize = (12, 2))
  sns.countplot(x = 'GlassType', data = glass_df)
  st.pyplot()
  
if 'Pie Chart' in plot_types:
  # plot pie chart
  st.subheader("Pie Chart")
  plt.figure(figsize = (12, 2))
  pie_data = glass_df['GlassType'].value_counts()
  plt.pie(pie_data, autopct = "%1.2f%%") #labels = pie_data.index
  st.pyplot()
if 'Correlation Heatmap' in plot_types:
  # plot correlation heatmap
  st.subheader("Correlation Heatmap")
  plt.figure(figsize = (12, 2))
  sns.heatmap(glass_df.corr(), annot = True)
  st.pyplot()
if 'Pair Plot' in plot_types:
  # plot pair plot
  st.subheader("Pair Plot")
  plt.figure(figsize = (12, 2))
  sns.pairplot(glass_df)
  st.pyplot()
# S2.1: Add 9 slider widgets for accepting user input for 9 features.
st.sidebar.subheader("Select the values:")
ri = st.sidebar.slider("Ri", float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider("Na", float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider("Mg", float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider("Al", float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider("Si", float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider("K", float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider("Ca", float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider("Ba", float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider("Fe", float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))

st.sidebar.subheader("Choose Classifier")

# Add a selectbox in the sidebar with label 'Classifier'.
classifier = st.sidebar.selectbox("Classifier",('Support Vector Machine', 'Random Forest Classifier'))

# S4.1: Implement SVM with hyperparameter tuning
# if classifier == 'Support Vector Machine', ask user to input the values of 'C','kernel' and 'gamma'.
if classifier == 'Support Vector Machine':
  st.sidebar.subheader("Model Hyperparameters")
  c=st.sidebar.number_input("C",1,100,step=1)
  gamma=st.sidebar.number_input("Gamma",1,100,step=1)
  kernel=st.sidebar.radio("Kernel",("linear","poly","rbf"))
    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
  if st.sidebar.button("Classify"):
    st.subheader("Support Vector Machine")
    svc = SVC(c = c, kernel = kernel, gamma = gamma)
    svc.fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    score = svc.score(X_test, y_test)
    glass_type = prediction(svc,ri,na,mg,al,si,k,ca,ba,fe)
    st.write("Type of Glass :", glass_type)
    st.write("Score : ", score)
    plot_confusion_matrix(svc,X_test,y_test)
    st.pyplot()
