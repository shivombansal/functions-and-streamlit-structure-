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
        
        # [Previous dashboard code remains the same until AI analysis section]
        
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
