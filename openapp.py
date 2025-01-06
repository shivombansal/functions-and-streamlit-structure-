import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_openai_analysis(data: pd.DataFrame, prompt: str) -> str:
    """Get analysis from OpenAI based on the data and prompt."""
    if not client.api_key:
        return "OpenAI API key not configured. Please check your .env file."
    
    try:
        data_str = data.to_string()

        # Create the ChatCompletion request
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are an expert manufacturing analyst. 
                Analyze the production data and provide insights in a clear, professional manner. 
                Focus on OEE (Overall Equipment Effectiveness), quality rates, and production metrics.
                When explaining combinations, consider the relationships between machines, molds, and operators."""},
                {"role": "user", "content": f"{prompt}\n\nData:\n{data_str}"}
            ],
            temperature=0.7,
            max_tokens=500
        )

        # Access the generated response
        response_message = completion.choices[0].message.content
        return response_message
    
    except Exception as e:
        return f"Error getting OpenAI analysis: {str(e)}"

def create_visualizations(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Create various visualizations for the data."""
    figures = {}
    
    # OEE Distribution
    figures['oee_dist'] = px.histogram(
        df, x='OEE_mean',
        title='Distribution of Average OEE',
        labels={'OEE_mean': 'Average OEE (%)'},
        nbins=20
    )
    
    # Top Performers Bubble Chart
    figures['top_performers'] = px.scatter(
        df.head(10),
        x='Quality_Rate', y='OEE_mean',
        size='Total Production_sum',
        color='Machine Name',
        hover_data=['Mold Name', 'Operator'],
        title='Top 10 Combinations: OEE vs Quality Rate',
        labels={
            'Quality_Rate': 'Quality Rate (%)',
            'OEE_mean': 'Average OEE (%)',
            'Total Production_sum': 'Total Production'
        }
    )
    
    return figures

def analyze_production_data(data: pd.DataFrame) -> pd.DataFrame:
    """Analyze production data to calculate metrics for machine-mold-operator combinations."""
    # Group by machine, mold, and operator
    grouped = data.groupby(['Machine Name', 'Mold Name', 'Operator'])
    
    # Calculate metrics for each combination
    combinations = grouped.agg({
        'OEE': ['mean', 'std', 'count'],
        'Good Part': ['sum'],
        'Bad Part (nos.)': ['sum'],
        'Total Production': ['sum']
    }).reset_index()
    
    # Flatten column names
    combinations.columns = [
        f'{col[0]}{"_" + col[1] if col[1] else ""}' 
        for col in combinations.columns
    ]
    
    # Calculate quality rate
    combinations['Quality_Rate'] = (
        combinations['Good Part_sum'] / 
        combinations['Total Production_sum'] * 100
    )
    
    # Sort by OEE mean in descending order
    combinations = combinations.sort_values('OEE_mean', ascending=False)
    
    return combinations

def main():
    # Set up Streamlit page
    st.set_page_config(page_title="Enhanced OEE Analysis Dashboard", layout="wide")
    
    # Get OpenAI API key from .env or Streamlit input
    openai_api_key = os.getenv('OPENAI_API_KEY')

    with st.sidebar:
        st.title("Configuration")
        
        # Check for API key in .env
        if not openai_api_key:
            st.warning("OpenAI API key not found in .env file.")
            openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
            if openai_api_key:
                st.success("API key set successfully.")
        else:
            st.success("OpenAI API key loaded from .env file.")

    # Main title
    st.title("Enhanced OEE Analysis Dashboard")
    
    # Load and analyze data
    try:
        data = pd.read_csv(r"C:\Users\Shivo\Desktop\RA BIZ TEK\Plastech\year_report.csv")
    except FileNotFoundError:
        st.error("Data file not found. Please check the file path.")
        return

    # Perform production data analysis
    df_combinations = analyze_production_data(data)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Dashboard", "AI Analysis", "Query Assistant"])
    
    # Tab 1: Dashboard
    with tab1:
        st.markdown("### Machine-Mold-Operator Performance Analysis")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            machines = ['All'] + sorted(df_combinations['Machine Name'].unique().tolist())
            selected_machine = st.selectbox('Select Machine:', machines, key='dash_machine')
        with col2:
            molds = ['All'] + sorted(df_combinations['Mold Name'].unique().tolist())
            selected_mold = st.selectbox('Select Mold:', molds, key='dash_mold')
        with col3:
            operators = ['All'] + sorted(df_combinations['Operator'].unique().tolist())
            selected_operator = st.selectbox('Select Operator:', operators, key='dash_operator')
        
        # Apply filters
        filtered_df = df_combinations.copy()
        if selected_machine != 'All':
            filtered_df = filtered_df[filtered_df['Machine Name'] == selected_machine]
        if selected_mold != 'All':
            filtered_df = filtered_df[filtered_df['Mold Name'] == selected_mold]
        if selected_operator != 'All':
            filtered_df = filtered_df[filtered_df['Operator'] == selected_operator]
        
        # Create and display visualizations
        figures = create_visualizations(filtered_df)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(figures['oee_dist'], use_container_width=True)
        with col2:
            st.plotly_chart(figures['top_performers'], use_container_width=True)
        
        # Display metrics and table
        st.markdown("### Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average OEE", f"{filtered_df['OEE_mean'].mean():.2f}%")
        with col2:
            st.metric("Best OEE", f"{filtered_df['OEE_mean'].max():.2f}%")
        with col3:
            st.metric("Total Combinations", len(filtered_df))
        with col4:
            st.metric("Average Quality Rate", f"{filtered_df['Quality_Rate'].mean():.2f}%")
        
        # Display detailed table
        st.markdown("### Detailed Analysis")
        display_columns = {
            'Machine Name': 'Machine',
            'Mold Name': 'Mold',
            'Operator': 'Operator',
            'OEE_mean': 'Avg OEE (%)',
            'OEE_std': 'OEE Std Dev',
            'OEE_count': 'Number of Runs',
            'Good Part_sum': 'Total Good Parts',
            'Bad Part (nos.)_sum': 'Total Bad Parts',
            'Total Production_sum': 'Total Production',
            'Quality_Rate': 'Quality Rate (%)'
        }
        filtered_df_display = filtered_df.rename(columns=display_columns)
        st.dataframe(
            filtered_df_display[display_columns.values()],
            hide_index=True,
            use_container_width=True
        )
    
    # Tab 2: AI Analysis
    with tab2:
        st.markdown("### AI-Powered Analysis")
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar to enable AI analysis.")
        else:
            # Top combinations analysis
            top_combinations = df_combinations.head(5)
            analysis_prompt = """Analyze the top 5 performing combinations of machine, mold, and operator. 
            Explain why these combinations might be performing well, considering their OEE, quality rates, 
            and production volumes. Provide specific insights and recommendations."""
            
            with st.spinner("Generating analysis..."):
                analysis = get_openai_analysis(top_combinations, analysis_prompt)
                st.markdown(analysis)
    
    # Tab 3: Query Assistant
    with tab3:
        st.markdown("### Query Assistant")
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar to enable the query assistant.")
        else:
            user_query = st.text_area("Ask a question about the production data:", 
                                    height=100,
                                    placeholder="Example: Which machine-mold combinations have the highest quality rates?")
            
            if st.button("Get Answer"):
                with st.spinner("Analyzing..."):
                    answer = get_openai_analysis(df_combinations, user_query)
                    st.markdown("### Answer")
                    st.markdown(answer)
                    
                    # Generate relevant visualization based on the query
                    st.markdown("### Related Visualization")
                    if "quality" in user_query.lower():
                        fig = px.scatter(
                            df_combinations.sort_values('Quality_Rate', ascending=False).head(10),
                            x='Machine Name',
                            y='Quality_Rate',
                            size='Total Production_sum',
                            color='OEE_mean',
                            hover_data=['Mold Name', 'Operator'],
                            title='Top 10 Combinations by Quality Rate'
                        )
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
