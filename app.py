import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import json
from datetime import datetime
import os
from manufacturing_analyzer import ManufacturingAnalyzer
from manufacturing_analyzer import PresentationFormatter
import streamlit as st
import pandas as pd
import plotly.express as px
import logging
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any
import json
from datetime import datetime
import time


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamlitManufacturingAnalyzer:
    def __init__(self):
        self.analyzer = ManufacturingAnalyzer()
        self.analyzer.initialize_model()
        self.formatter = PresentationFormatter(self.analyzer)
        self.chat = None
        self.data = None
        # Store model reference directly for easy access
        self.model = self.analyzer.model
        
    def initialize_chat(self):
        """Initialize or reset the chat session"""
        try:
            if not self.model:
                # Re-initialize model if it's not available
                self.analyzer.initialize_model()
                self.model = self.analyzer.model
            self.chat = self.model.start_chat()
            return True
        except Exception as e:
            logger.error(f"Error initializing chat: {str(e)}")
            return False
    
    def run_analysis(self, data, analysis_prompt):
        """Run the initial analysis with formatting."""
        try:
            if not self.chat:
                self.initialize_chat()
                
            message = {
                "parts": [
                    {
                        "text": f"{analysis_prompt}\n\nData: {json.dumps(data)}"
                    }
                ]
            }
            response = self.chat.send_message(message)
            results = self.analyzer.process_response(response)
            
            # Format the results
            formatted_results = self.formatter.format_analysis_results(results)
            return formatted_results
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            return None
            
    def process_query(self, query):
        """Process follow-up queries with formatting."""
        try:
            if not self.chat:
                self.initialize_chat()
                
            message = {
                "parts": [
                    {
                        "text": f"Based on the manufacturing data previously analyzed, please answer this query: {query}"
                    }
                ]
            }
            response = self.chat.send_message(message)
            results = self.analyzer.process_response(response)
            
            # Format the results
            formatted_results = self.formatter.format_analysis_results(results)
            return formatted_results
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return None

def main():
    st.set_page_config(page_title="Manufacturing Analysis Dashboard", layout="wide")
    
    st.title("Manufacturing Analysis Dashboard")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StreamlitManufacturingAnalyzer()
        st.session_state.analysis_results = None
        st.session_state.chat_history = []
        st.session_state.show_query = False
    
    # File upload
    uploaded_file = st.file_uploader("Upload manufacturing data (CSV)", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.analyzer.data = df.to_dict('records')
            
            st.success("File uploaded successfully!")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Initial Analysis Section
            st.subheader("Initial Analysis")
            if st.button("Run Initial Analysis"):
                with st.spinner("Running analysis..."):
                    # Use the actual uploaded data
                    analysis_prompt = """Please analyze the manufacturing data and provide insights using these functions:
                    1. Calculate machine performance using calculateAverageOEE
                    2. Analyze downtime using calculateAverageUnplannedDowntime
                    3. Predict future performance using predictOEEForNextMonth
                    Please provide comprehensive insights from the analysis in a tabular form."""
                    
                    # Initialize new chat session
                    st.session_state.analyzer.initialize_chat()
                    
                    # Run analysis with actual data
                    results = st.session_state.analyzer.run_analysis(
                        st.session_state.analyzer.data,
                        analysis_prompt
                    )
                    
                    if results:
                        st.session_state.analysis_results = results
                        st.session_state.show_query = True  # Enable query section
                        display_results(results)
                        
            # Query Section - Always show if analysis has been run
            if st.session_state.show_query:
                st.subheader("Ask Questions About the Data")
                query = st.text_input("Enter your query about the manufacturing data:")
                
                if st.button("Submit Query"):
                    with st.spinner("Processing query..."):
                        query_results = st.session_state.analyzer.process_query(query)
                        
                        if query_results:
                            # Add to chat history
                            st.session_state.chat_history.append({
                                "query": query,
                                "response": query_results
                            })
                            
                            # Display chat history
                            display_chat_history()
                
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a file containing data.")
        except pd.errors.ParserError:
            st.error("Error parsing the CSV file. Please make sure it's properly formatted.")
        except KeyError as e:
            st.error(f"Required column missing in the data: {str(e)}")
        except ValueError as e:
            st.error(f"Invalid data format: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logging.error(f"Unexpected error in main: {str(e)}", exc_info=True)

def display_chat_history():
    """Display the chat history with queries and responses"""
    st.subheader("Previous Queries")
    
    for interaction in st.session_state.chat_history:
        st.markdown("---")
        st.write("**Your Query:**")
        st.write(interaction["query"])
        st.write("**Response:**")
        display_results(interaction["response"])

def display_results(results):
    """Display analysis results in a structured format with improved error handling"""
    try:
        # Initialize analysis prompt with clear function specifications
        analysis_prompt = """Please analyze the manufacturing data and provide insights in the following format:

        1. Overall Performance Summary:
           - Use calculateAverageOEE to compute OEE metrics
           - Present key statistics (mean, min, max)

        2. Downtime Analysis:
           - Use calculateAverageUnplannedDowntime for downtime metrics
           - Highlight critical issues

        3. Future Predictions:
           - Use predictOEEForNextMonth for forecasting
           - Include confidence intervals

        Please structure your response as a clear report with sections."""

        # Validate chat session
        if not st.session_state.analyzer.chat:
            st.session_state.analyzer.initialize_chat()
        
        # Format the data and ensure it's properly structured
        if isinstance(results, str):
            data = {"text": results}
        else:
            data = results

        # Create message with proper structure
        message = {
            "parts": [
                {
                    "text": f"{analysis_prompt}\n\nAnalysis Results: {json.dumps(data)}"
                }
            ]
        }
        
        try:
            # Send message with retry logic
            max_retries = 3
            retry_delay = 1  # Starting delay in seconds
            
            for attempt in range(max_retries):
                try:
                    response = st.session_state.analyzer.chat.send_message(message)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
            
            # Display the formatted analysis
            st.markdown("## 📊 Manufacturing Analysis Report")
            
            # Process response sections
            if hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'text'):
                        # Split and format text sections
                        sections = part.text.split('\n\n')
                        for section in sections:
                            if section.strip():
                                if section[0].isdigit():
                                    st.markdown(f"### {section}")
                                else:
                                    st.write(section)
                    
                    elif hasattr(part, 'function_call'):
                        # Handle function calls with proper error checking
                        fn = part.function_call
                        if hasattr(fn, 'name') and hasattr(fn, 'args'):
                            try:
                                # Validate function arguments
                                args = json.loads(fn.args) if isinstance(fn.args, str) else fn.args
                                
                                if fn.name == 'calculateAverageOEE':
                                    display_oee_metrics(args)
                                elif fn.name == 'calculateAverageUnplannedDowntime':
                                    display_downtime_metrics(args)
                                elif fn.name == 'predictOEEForNextMonth':
                                    display_predictions(args)
                            except json.JSONDecodeError:
                                st.error(f"Invalid function arguments format for {fn.name}")
                            except Exception as e:
                                st.error(f"Error processing function {fn.name}: {str(e)}")

            # Add export functionality
            if st.button("📥 Export Analysis Report"):
                export_report(response)

        except Exception as e:
            st.error(f"Error communicating with the model: {str(e)}")
            logger.error(f"Model communication error: {str(e)}")
            
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        logger.error(f"Display error: {str(e)}", exc_info=True)

def display_oee_metrics(args):
    """Display OEE metrics with visualizations"""
    try:
        data = args.get('data', [])
        df = pd.DataFrame(data)
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_oee = df['OEE'].mean()
            st.metric("Average OEE", f"{avg_oee:.2f}%")
            
        with col2:
            max_oee = df['OEE'].max()
            max_machine = df.loc[df['OEE'].idxmax(), 'Machine Name']
            st.metric("Best Performer", f"{max_machine}\n({max_oee:.2f}%)")
            
        with col3:
            min_oee = df['OEE'].min()
            min_machine = df.loc[df['OEE'].idxmin(), 'Machine Name']
            st.metric("Needs Improvement", f"{min_machine}\n({min_oee:.2f}%)")
        
        # Create performance chart
        fig = px.bar(df,
                    x='Machine Name',
                    y='OEE',
                    title='OEE Performance by Machine',
                    labels={'OEE': 'OEE (%)', 'Machine Name': 'Machine'},
                    color='OEE',
                    color_continuous_scale='viridis')
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying OEE metrics: {str(e)}")

def display_downtime_metrics(args):
    """Display downtime metrics with visualizations"""
    try:
        data = args.get('data', [])
        df = pd.DataFrame(data)
        
        fig = px.bar(df,
                    x='Machine Name',
                    y='Unplan D/T (Min.)',
                    title='Unplanned Downtime by Machine',
                    labels={'Unplan D/T (Min.)': 'Minutes', 'Machine Name': 'Machine'},
                    color='Unplan D/T (Min.)',
                    color_continuous_scale='reds')
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying downtime metrics: {str(e)}")

def display_predictions(args):
    """Display prediction results with confidence intervals"""
    try:
        data = args.get('data', {})
        prediction = data.get('predicted_value', 0)
        confidence = data.get('confidence_interval', [0, 0])
        
        st.markdown("### Predictions for Next Month")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted OEE", f"{prediction:.2f}%")
        with col2:
            st.metric("Lower Bound", f"{confidence[0]:.2f}%")
        with col3:
            st.metric("Upper Bound", f"{confidence[1]:.2f}%")
    except Exception as e:
        st.error(f"Error displaying predictions: {str(e)}")

def export_report(response):
    """Export the analysis report as a text file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_text = "\n\n".join([
            part.text for part in response.parts 
            if hasattr(part, 'text')
        ])
        
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name=f"manufacturing_analysis_{timestamp}.txt",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"Error exporting report: {str(e)}")
                    
if __name__ == "__main__":
    main()