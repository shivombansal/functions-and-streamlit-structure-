import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict
from openai import OpenAI
import httpx
import os
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

def get_openai_client():
    """Create and return an OpenAI client configured with Portkey gateway."""
    try:
        client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            base_url=PORTKEY_GATEWAY_URL,
            default_headers=createHeaders(
                provider="openai",
                api_key=st.secrets["PORTKEY_API_KEY"]
            ),
            http_client=httpx.Client()
        )
        return client
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

def get_openai_analysis(data: pd.DataFrame, prompt: str) -> str:
    """Get OpenAI analysis."""
    client = get_openai_client()
    if not client:
        return "OpenAI client initialization failed. Please check your configuration."
    
    try:
        data_str = data.to_string()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are an expert manufacturing analyst. 
                Analyze the production data and provide insights in a clear, professional manner."""},
                {"role": "user", "content": f"{prompt}\n\nData:\n{data_str}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error getting OpenAI analysis: {str(e)}"


def create_visualizations(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Create various visualizations for the data."""
    figures = {}
    
    # Trend Line - OEE vs Total Production
    figures['trend_line'] = px.strip(
        df.head(15),
        x='Operator',
        y='OEE_mean',
        title='Trend Line: OEE by top-15 Operator',
        labels={
            'Operator': 'Operator',
            'OEE_mean': 'Average OEE (%)'
        }
    )

    figures['avg_oee_top10'] = px.bar(
        df.head(15),
        x='Machine Name',
        y='OEE_mean',
        color='Operator',
        barmode='group',
        hover_data=['Mold Name', 'Total Production_sum'],
        title='Average OEE vs Top 15 Combinations',
        labels={
            'OEE_mean': 'Average OEE (%)',
            'Machine Name': 'Machine',
            'Operator': 'Operator'
        }
    )
    
    # Update layout for thicker bars and better spacing
    figures['avg_oee_top10'].update_layout(
        yaxis_title="OEE (%)",
        xaxis_title="Machine",
        bargap=0.01,
        bargroupgap=0.5,
        yaxis=dict(
            gridwidth=1,
            zeroline=False
        ),
        xaxis=dict(
            zeroline=False
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1
        )
    )
    
    figures['avg_oee_top10'].update_traces(width=0.2)
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

    try:
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file")
            return
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Perform production data analysis
    df_combinations = analyze_production_data(data)
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Dashboard", "Ask Plastech AI"])
    
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
            st.plotly_chart(figures['trend_line'], use_container_width=True)
        with col2:
            st.plotly_chart(figures['avg_oee_top10'], use_container_width=True)
        
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
    
        st.markdown("### Plastech AI Analysis")
        
        if st.button("Generate AI Analysis", key="generate_analysis"):
            top_combinations = filtered_df
            analysis_prompt = """
            Analyze the performance of all machine-mold-operator combinations in the dataset. 
            Provide:
            1. Key overall insights on the dataset, focusing on OEE, quality rates, and production volumes.
            2. Patterns or trends observed in the data across machines, molds, and operators.
            3. Suggestions to improve underperforming combinations based on the insights.
            4. Recommendations to enhance overall production efficiency and quality.
            """
            
            with st.spinner("Generating analysis..."):
                analysis = get_openai_analysis(top_combinations, analysis_prompt)
                st.markdown(analysis)
    
    # Tab 2: Query Assistant
    with tab2:
        st.markdown("### Ask Plastech AI")
        user_query = st.text_area("Ask a question about the production data:", 
                                height=100,
                                placeholder="Example: Which machine-mold combinations have the highest quality rates?")
        
        if st.button("Get Answer"):
            with st.spinner("Analyzing..."):
                answer = get_openai_analysis(df_combinations, user_query)
                st.markdown("### Answer")
                st.markdown(answer)
                

if __name__ == "__main__":
    main()
