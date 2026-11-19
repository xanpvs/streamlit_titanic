import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Titanic Data Analysis",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Titanic Data Analysis Dashboard")
st.write("Interactive analysis of Titanic passengers")

AGE_BINS = [-1, 12, 18, 35, 60, float("inf")]
AGE_LABELS = [
    "Child",
    "Teen",
    "Young Adult",
    "Adult",
    "Older Adult"
]


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    return pd.read_csv(url)


@st.cache_data
def preprocess_data(raw_df):
    data = raw_df.copy()

    age_median = data["age"].median()
    data["age"] = data["age"].fillna(age_median)

    embarked_mode = data["embarked"].mode()[0]
    data["embarked"] = data["embarked"].fillna(embarked_mode)

    data["age_minmax"] = (
        (data["age"] - data["age"].min()) /
        (data["age"].max() - data["age"].min())
    )

    data["fare_minmax"] = (
        (data["fare"] - data["fare"].min()) /
        (data["fare"].max() - data["fare"].min())
    )

    data["age_group"] = pd.cut(
        data["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS
    )

    df_encoded = data.copy()
    df_encoded["sex_code"] = df_encoded["sex"].map({"female": 0, "male": 1})
    df_encoded["embarked_code"] = df_encoded["embarked"].map({"C": 0, "Q": 1, "S": 2})

    return data, df_encoded


raw_df = load_data()
df, df_encoded = preprocess_data(raw_df)

if "page" not in st.session_state:
    st.session_state.page = "1. Data Inspection"

st.sidebar.header("Go to page")

if st.sidebar.button("1. Data Inspection", use_container_width=True):
    st.session_state.page = "1. Data Inspection"

if st.sidebar.button("2. Cleaning & Transformation", use_container_width=True):
    st.session_state.page = "2. Cleaning & Transformation"

if st.sidebar.button("3. Interactive Dashboard & Findings", use_container_width=True):
    st.session_state.page = "3. Interactive Dashboard & Findings"

page = st.session_state.page

if page == "1. Data Inspection":
    st.header("1. Data Inspection")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of Rows", raw_df.shape[0])

    with col2:
        st.metric("Number of Columns", raw_df.shape[1])

    st.subheader("Column Names")
    st.write(raw_df.columns.tolist())

    st.subheader("Data Types")
    st.write(raw_df.dtypes.astype(str))

    st.subheader("First Five Rows")
    st.dataframe(raw_df.head())

    st.subheader("Missing Values Before Cleaning")
    missing_before = raw_df.isnull().sum()
    st.dataframe(missing_before.rename("Missing Count"))

    st.subheader("Example of Row Selection")
    st.dataframe(raw_df.iloc[3:7, 5:8])

elif page == "2. Cleaning & Transformation":
    st.header("2. Missing Value Handling")

    st.subheader("Missing Values After Cleaning")
    missing_after = df.isnull().sum()
    st.dataframe(missing_after.rename("Missing Count"))

    st.header("3. Feature Transformation")

    st.subheader("Transformed Features")
    st.dataframe(
        df[
            ["age", "age_minmax", "fare", "fare_minmax", "age_group"]
        ].head(10)
    )

    st.header("4. Dummy Coding")

    st.write("Before encoding")
    st.dataframe(
        df[["sex", "embarked"]].head()
    )

    st.write("After encoding (single column per feature)")
    st.caption("sex_code: female -> 0, male -> 1 | embarked_code: C -> 0, Q -> 1, S -> 2")
    st.dataframe(
        df_encoded[["sex_code", "embarked_code"]].head()
    )

elif page == "3. Interactive Dashboard & Findings":
    st.header("5. Interactive Dashboard")

    st.sidebar.header("Passenger Filters")

    class_options = ["All"] + [str(c) for c in sorted(df["pclass"].dropna().unique())]
    selected_class = st.sidebar.selectbox(
        "Passenger Class",
        class_options
    )

    sex_options = ["All"] + sorted(df["sex"].dropna().unique())
    selected_sex = st.sidebar.selectbox(
        "Sex",
        sex_options
    )

    age_options = ["All"] + AGE_LABELS
    selected_age = st.sidebar.selectbox(
        "Age Group",
        age_options
    )

    filtered_df = df.copy()
    if selected_class != "All":
        filtered_df = filtered_df[filtered_df["pclass"] == int(selected_class)]
    if selected_sex != "All":
        filtered_df = filtered_df[filtered_df["sex"] == selected_sex]
    if selected_age != "All":
        filtered_df = filtered_df[filtered_df["age_group"] == selected_age]

    st.subheader("Key Metrics")

    col1, col2, col3 = st.columns(3)

    total_passengers = len(filtered_df)

    survival_rate = (
        filtered_df["survived"].mean() * 100
        if total_passengers > 0 else 0
    )

    average_fare = (
        filtered_df["fare"].mean()
        if total_passengers > 0 else 0
    )

    col1.metric("Total Passengers", total_passengers)
    col2.metric("Survival Rate", f"{survival_rate:.2f}%")
    col3.metric("Average Fare", f"{average_fare:.2f}")

    st.subheader("Survival Rate by Age Group")
    if total_passengers > 0:
        age_chart = (
            filtered_df.groupby(
                "age_group",
                observed=False
            )["survived"].mean() * 100
        )
        st.bar_chart(age_chart)

    st.subheader("Survival Rate by Sex")
    if total_passengers > 0:
        sex_chart = (
            filtered_df.groupby("sex")["survived"].mean() * 100
        )
        st.bar_chart(sex_chart)

    st.subheader("Survival Rate by Passenger Class")
    if total_passengers > 0:
        class_chart = (
            filtered_df.groupby("pclass")["survived"].mean() * 100
        )
        st.bar_chart(class_chart)

    st.header("6. Findings")
    if total_passengers > 0:
        st.write(
            "The charts show the observed survival rates "
            "for the selected passenger groups. "
            "These results describe associations and "
            "do not establish causation."
        )
    else:
        st.warning("No passengers match the selected filters.")