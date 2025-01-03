import streamlit as st
import pandas as pd

# Load the dataset
@st.cache_data
def load_data():
    # Replace "your_data.csv" with the path to your dataset
    df = pd.read_csv("year_report.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['OEE'] = pd.to_numeric(df['OEE'], errors='coerce')
    return df

# Load data
data = load_data()

# Sidebar for user selection
st.sidebar.header("Filter Options")

# Dropdowns for Machine, Mold, and Operator
machine_name = st.sidebar.selectbox("Select Machine", ["All"] + sorted(data['Machine Name'].unique().tolist()))
mold_name = st.sidebar.selectbox("Select Mold", ["All"] + sorted(data['Mold Name'].unique().tolist()))
operator_name = st.sidebar.selectbox("Select Operator", ["All"] + sorted(data['Operator'].unique().tolist()))

# Filter data based on selections
filtered_data = data.copy()

if machine_name != "All":
    filtered_data = filtered_data[filtered_data['Machine Name'] == machine_name]

if mold_name != "All":
    filtered_data = filtered_data[filtered_data['Mold Name'] == mold_name]

if operator_name != "All":
    filtered_data = filtered_data[filtered_data['Operator'] == operator_name]

# Display summary tables
if operator_name == "All":
    st.write("### Operator Efficiency Summary")
    operator_summary = (
        data.groupby('Operator')
        .agg(Total_Good_Parts=('Good Part', 'sum'),
             Average_OEE=('OEE', 'mean'))
        .reset_index()
        .sort_values(by='Average_OEE', ascending=False)
    )
    st.dataframe(operator_summary)

if machine_name == "All":
    st.write("### Machine Efficiency Summary")
    machine_summary = (
        data.groupby(['Machine Name', 'Mold Name'])
        .agg(Average_OEE=('OEE', 'mean'))
        .reset_index()
        .sort_values(by='Average_OEE', ascending=False)
    )
    st.dataframe(machine_summary)

if mold_name == "All":
    st.write("### Mold Efficiency Summary")
    mold_summary = (
        data.groupby('Mold Name')
        .agg(Average_OEE=('OEE', 'mean'))
        .reset_index()
        .sort_values(by='Average_OEE', ascending=False)
    )
    st.dataframe(mold_summary)

# Display filtered data
st.write("### Filtered Data")
st.dataframe(filtered_data)
