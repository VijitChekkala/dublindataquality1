# Updated import statement for ydata-profiling
from ydata_profiling import ProfileReport

# You don't need to specify a specific Pydantic version unless necessary
# Use a recent stable version of Pydantic for compatibility

# Your remaining code remains unchanged
import streamlit as st
import numpy as np
from scipy import stats
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import json
import plotly.figure_factory as ff
import statsmodels.api as sm
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
import sympy as smp
import seaborn as sns
from pydantic_settings import BaseSettings


def pandas_profiling_report(df):
    df_report = ProfileReport(df, explorative=True)
    return df_report

def read_csv(source_data):
    df = pd.read_csv(source_data)
    return df 

def read_excel(source_data):
    df = pd.read_excel(source_data)
    return df

def OLS(df, S1):
    train = df.drop([S1], axis=1)
    test = df[S1]
    constant = sm.add_constant(train)
    model = sm.OLS(list(test), constant)
    result = model.fit()
    pred = result.predict()
    return pred

def main():
    df = None
    with st.sidebar.header("Source Data Selection"):
        selection = ["csv", 'excel']
        selected_data = st.sidebar.selectbox("Please select your dataset format:", selection)
        if selected_data is not None:
            if selected_data == "csv":
                st.sidebar.write("Select Dataset")
                source_data = st.sidebar.file_uploader("Upload/select source (.csv) data", type=["csv"])
                if source_data is not None: 
                    df = read_csv(source_data)
            elif selected_data == "excel":
                st.sidebar.write("Select Dataset")
                source_data = st.sidebar.file_uploader("Upload/select source (.xlsx) data", type=["xlsx"])
                if source_data is not None:
                    df = read_excel(source_data)
    
    st.header("Dataset")
        
    if df is not None:
        user_choices = ['Dataset Sample', "Data Quality"]
        selected_choices = st.sidebar.selectbox("Please select your choice:", user_choices)
        
        if selected_choices is not None:
            if selected_choices == "Dataset Sample":
                st.info("Select dataset has " + str(df.shape[0]) + " rows and " + str(df.shape[1]) + " columns.")
                st.write(df)  
            elif selected_choices == "Data Quality":
                box = ["Overview", "Score", "Data types", "Descriptive statistics", "Missing values", 
                       "Duplicate records", "Correlation", "Outliers", "Data distribution"]
                selection = st.selectbox("Data Quality Selection", box, key=f"MyKey{4}") 
                if selection is not None:
                    if selection == "Overview":
                        df_report = pandas_profiling_report(df)
                        st.write("Profiling")
                        st_profile_report(df_report)
                    elif selection == "Data types":
                        types = pd.DataFrame(df.dtypes)
                        a = types.astype(str)
                        st.dataframe(a)
                    elif selection == "Descriptive statistics":
                        types = pd.DataFrame(df.describe()).T
                        a = types.astype(str)
                        st.table(a)
                    elif selection == "Missing values":
                        df.replace(0, np.nan, inplace=True)
                        types = pd.DataFrame(df.isnull().sum())
                        a = types.astype(str)
                        st.write(a)
                        box = df.keys()
                        se = st.selectbox("Show missing values", box, key=f"MyKey{5}")
                        for i in box:
                            if se == i:
                                st.write(df[pd.isnull(df[i])])
                    elif selection == "Duplicate records":
                        types = df[df.duplicated()]
                        a = types.astype(str)
                        st.write("The number of duplicated rows is ", len(types))
                        st.write(a)
                    elif selection == "Outliers":
                        fig = plt.figure(figsize=(4, 3))
                        box = df.keys()
                        se = st.selectbox("Select which column you want to check", box, key=f"MyKey{5}")
                        for i in box:
                            if se == i and df[i].dtypes != object:
                                sns.boxplot(df[i])
                                st.pyplot(fig)
                    elif selection == "Data distribution":
                        box = df.keys()
                        se = st.selectbox("Select which column you want to check", box, key=f"MyKey{6}")
                        for i in box:
                            if se == i and df[i].dtypes != object:
                                fig = plt.figure(figsize=(4, 3))
                                sns.histplot(data=df, x=i, binwidth=3)
                                st.pyplot(fig)
                    elif selection == "Correlation":
                        fig, ax = plt.subplots()
                        sns.heatmap(df.corr(), annot=True, ax=ax)
                        st.pyplot(fig)
                    elif selection == "Score":
                        df.replace(0, np.nan, inplace=True)
                        x = []
                        y = max(df.isnull().sum())
                        z = df.duplicated().sum()
                        box = df.keys()
                        for i in box:
                            if df[i].dtypes != object:
                                x.append(len(df[(np.abs(stats.zscore(df[i])) >= 3)]))
                        error = sum(x) + y + z
                        st.write("Overall, the score of data is ", 1 - error / len(df))
       
    else:
        st.error("Please select your data to start")

main()
