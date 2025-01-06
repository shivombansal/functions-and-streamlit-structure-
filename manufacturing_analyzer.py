from dotenv import load_dotenv
import os
import logging
import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from google.api_core import retry
import asyncio
from functools import wraps
import concurrent.futures
import threading
import time

class TimeoutError(Exception):
    pass

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=120):
    """
    Run a function with a timeout using threads.
    Returns the function result or raises TimeoutError.
    """
    result = []
    error = []
    
    def worker():
        try:
            ret = func(*args, **kwargs)
            result.append(ret)
        except Exception as e:
            error.append(e)
            
    thread = threading.Thread(target=worker)
    thread.daemon = True
    
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        thread.join(1)  # Give it a second to clean up
        raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")
        
    if error:
        raise error[0]
        
    if not result:
        raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")
        
    return result[0]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Helper class for performing data analysis calculations."""
    
    @staticmethod
    def calculate_metrics_by_group(df: pd.DataFrame, group_column: str, metric_column: str) -> Dict[str, Dict[str, float]]:
        """Calculate multiple metrics (mean, min, max, std) grouped by a column."""
        try:
            grouped = df.groupby(group_column)[metric_column].agg(['mean', 'min', 'max', 'std']).round(2)
            return grouped.to_dict('index')
        except Exception as e:
            logger.error(f"Error in calculate_metrics_by_group: {str(e)}")
            raise

    @staticmethod
    def predict_next_month(df: pd.DataFrame, value_column: str, timestamp_column: str) -> Dict[str, float]:
        """Simple prediction for next month using moving average."""
        try:
            # Convert timestamp with explicit format
            df['datetime'] = pd.to_datetime(df[timestamp_column], format='%d-%m-%y %H:%M')
            df = df.sort_values('datetime')
            
            # Calculate 3-month moving average
            moving_avg = df[value_column].rolling(window=3).mean().iloc[-1]
            # Calculate trend
            trend = df[value_column].diff().mean()
            
            prediction = moving_avg + trend
            
            return {
                'predicted_value': round(prediction, 2),
                'confidence_interval': [
                    round(prediction - prediction * 0.1, 2),  # Lower bound
                    round(prediction + prediction * 0.1, 2)   # Upper bound
                ]
            }
        except Exception as e:
            logger.error(f"Error in predict_next_month: {str(e)}")
            raise

class ManufacturingAnalyzer:
    def __init__(self):
        self._load_environment()
        self._configure_genai()
        self._setup_tools()
        self.data_analyzer = DataAnalyzer()
        
    def _load_environment(self) -> None:
        """Load environment variables safely."""
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

    def _configure_genai(self) -> None:
        """Configure Gemini AI with API key and generation settings."""
        genai.configure(api_key=self.api_key)
        self.generation_config = {
            "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

    def _setup_tools(self) -> None:
        """Set up all analysis tools with proper function declarations following Gemini API requirements."""
        try:
            self.tools = {
                "function_declarations": [
                    {
                        "name": "calculateAverageOEE",
                        "description": "Calculates average Overall Equipment Effectiveness (OEE) metrics grouped by machine. Returns mean, minimum, maximum, and standard deviation of OEE values for each machine.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "array",
                                    "description": "Array of manufacturing data records",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "Machine Name": {"type": "string"},
                                            "OEE": {"type": "number"}
                                        }
                                    }
                                },
                                "machine_column": {
                                    "type": "string",
                                    "description": "Name of the column containing machine identifiers, e.g., 'Machine Name'"
                                },
                                "oee_column": {
                                    "type": "string",
                                    "description": "Name of the column containing OEE values, e.g., 'OEE'"
                                }
                            },
                            "required": ["data", "machine_column", "oee_column"]
                        }
                    },
                    {
                        "name": "calculateAverageUnplannedDowntime",
                        "description": "Calculates average unplanned downtime metrics grouped by machine. Returns mean, minimum, maximum, and standard deviation of downtime values for each machine.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "array",
                                    "description": "Array of manufacturing data records",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "Machine Name": {"type": "string"},
                                            "Unplan D/T (Min.)": {"type": "number"}
                                        }
                                    }
                                },
                                "machine_column": {
                                    "type": "string",
                                    "description": "Name of the column containing machine identifiers, e.g., 'Machine Name'"
                                },
                                "downtime_column": {
                                    "type": "string",
                                    "description": "Name of the column containing unplanned downtime values in minutes, e.g., 'Unplan D/T (Min.)'"
                                }
                            },
                            "required": ["data", "machine_column", "downtime_column"]
                        }
                    },
                    {
                        "name": "predictOEEForNextMonth",
                        "description": "Predicts OEE values for the next month using historical data trends. Returns predicted value and confidence interval based on 3-month moving average and trend analysis.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "array",
                                    "description": "Array of historical manufacturing data records",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "Timestamp": {"type": "string"},
                                            "OEE": {"type": "number"}
                                        }
                                    }
                                },
                                "oee_column": {
                                    "type": "string",
                                    "description": "Name of the column containing OEE values, e.g., 'OEE'"
                                },
                                "timestamp_column": {
                                    "type": "string",
                                    "description": "Name of the column containing timestamp information, e.g., 'Timestamp'"
                                }
                            },
                            "required": ["data", "oee_column", "timestamp_column"]
                        }
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error setting up tools: {str(e)}")
            raise

    def calculateAverageOEE(self, data, machine_column, oee_column):
        """Calculate and format OEE data."""
        try:
            # Convert the input data to a pandas DataFrame
            df = pd.DataFrame(data)
            
            # Calculate metrics for each machine
            results = df.groupby(machine_column)[oee_column].agg({
                'mean': 'mean',
                'min': 'min',
                'max': 'max',
                'std': 'std'
            }).round(2)
            
            return {
                "status": "success",
                "data": results.to_dict('index')
            }
        except Exception as e:
            logger.error(f"Error in calculateAverageOEE: {str(e)}")
            return {"status": "error", "message": str(e)}

    def calculateAverageUnplannedDowntime(self, data: List[Dict], machine_column: str, downtime_column: str) -> Dict[str, Any]:
        """Calculate average unplanned downtime by machine."""
        try:
            df = pd.DataFrame(data)
            results = self.data_analyzer.calculate_metrics_by_group(df, machine_column, downtime_column)
            return {
                "status": "success",
                "data": results
            }
        except Exception as e:
            logger.error(f"Error in calculateAverageUnplannedDowntime: {str(e)}")
            return {"status": "error", "message": str(e)}

    def predictOEEForNextMonth(self, data: List[Dict], oee_column: str, timestamp_column: str) -> Dict[str, Any]:
        """Predict OEE values for the next month."""
        try:
            df = pd.DataFrame(data)
            prediction = self.data_analyzer.predict_next_month(df, oee_column, timestamp_column)
            return {
                "status": "success",
                "data": prediction
            }
        except Exception as e:
            logger.error(f"Error in predictOEEForNextMonth: {str(e)}")
            return {"status": "error", "message": str(e)}

    def initialize_model(self) -> None:
        """Initialize the Gemini model."""
        try:
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=self.generation_config,
                tools=self.tools
            )
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def process_response(self, response: Any) -> Dict[str, Any]:
        """Process and log the response from the model."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'function_calls': [],
            'analysis_results': []
        }
        
        try:
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fn = part.function_call
                    call_info = {
                        'name': fn.name,
                        'args': fn.args
                    }
                    results['function_calls'].append(call_info)
                    logger.info(f"Function called: {fn.name}")
                    
                    if hasattr(self, fn.name):
                        analysis_result = getattr(self, fn.name)(**fn.args)
                        results['analysis_results'].append({
                            'function': fn.name,
                            'result': analysis_result
                        })
                else:
                    text = part.text if hasattr(part, 'text') else str(part)
                    logger.info(f"Response text: {text}")
                    results['analysis_results'].append({
                        'type': 'text',
                        'content': text
                    })
                    
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            results['error'] = str(e)
        
        return results


class PresentationFormatter:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def format_analysis_results(self, results):
        """Format the analysis or query results."""
        try:
            # If results is already a string, return it directly
            if isinstance(results, str):
                return {'analysis_results': [{'type': 'text', 'content': results}]}
                
            # If results is None or empty, return an error message
            if not results:
                return {'analysis_results': [{'type': 'text', 'content': 'No results available'}]}
                
            # Ensure we have a properly structured results object
            if not isinstance(results, dict):
                return {'analysis_results': [{'type': 'text', 'content': str(results)}]}
                
            # Return the results directly if they're already in the correct format
            if 'analysis_results' in results:
                return results
                
            # Convert simple results into the expected format
            return {'analysis_results': [{'type': 'text', 'content': str(results)}]}
            
        except Exception as e:
            logger.error(f"Error in format_analysis_results: {str(e)}")
            return {'analysis_results': [{'type': 'text', 'content': f"Error formatting results: {str(e)}"}]}

    def _convert_results_to_string(self, results):
        """Convert the analysis results dictionary to a readable string format."""
        try:
            output = []
            
            # Handle case where results is already a string
            if isinstance(results, str):
                return results
                
            # Handle case where results is a dictionary
            if isinstance(results, dict):
                analysis_results = results.get('analysis_results', [])
                
                for item in analysis_results:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            output.append(f"Analysis: {item.get('content', '')}")
                        
                        elif 'function' in item and isinstance(item.get('result'), dict):
                            result = item['result']
                            if result.get('status') == 'success':
                                output.append(f"\nFunction: {item['function']}")
                                output.append("Results:")
                                
                                data = result.get('data', {})
                                if isinstance(data, dict):
                                    for key, value in data.items():
                                        if isinstance(value, dict):
                                            output.append(f"\n{key}:")
                                            for metric, val in value.items():
                                                output.append(f"  {metric}: {val}")
                                        else:
                                            output.append(f"{key}: {value}")
            
            return "\n".join(output) if output else "No results to display"
            
        except Exception as e:
            logger.error(f"Error in _convert_results_to_string: {str(e)}")
            return f"Error converting results to string: {str(e)}"
            
    def _convert_results_to_string(self, results):
        """Convert the analysis results dictionary to a readable string format."""
        try:
            output = []
            
            for item in results.get('analysis_results', []):
                if item.get('type') == 'text':
                    output.append(f"Analysis: {item['content']}")
                
                elif 'function' in item and item['result']['status'] == 'success':
                    output.append(f"\nFunction: {item['function']}")
                    output.append("Results:")
                    
                    data = item['result'].get('data', {})
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, dict):
                                output.append(f"\n{key}:")
                                for metric, val in value.items():
                                    output.append(f"  {metric}: {val}")
                            else:
                                output.append(f"{key}: {value}")
                                
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Error in _convert_results_to_string: {str(e)}")
            raise Exception(f"Failed to convert results to string: {str(e)}")
            
    def _convert_results_to_string(self, results):
        """Convert the analysis results dictionary to a readable string format."""
        output = []
        
        for item in results.get('analysis_results', []):
            if item.get('type') == 'text':
                output.append(f"Analysis: {item['content']}")
            
            elif 'function' in item and item['result']['status'] == 'success':
                output.append(f"\nFunction: {item['function']}")
                output.append("Results:")
                
                data = item['result'].get('data', {})
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            output.append(f"\n{key}:")
                            for metric, val in value.items():
                                output.append(f"  {metric}: {val}")
                        else:
                            output.append(f"{key}: {value}")
                            
        return "\n".join(output)
    
    
def main():
    """Main execution function with enhanced error handling and result management."""
    analyzer = None
    try:
        # Initialize analyzer
        analyzer = ManufacturingAnalyzer()
        analyzer.initialize_model()

        # Read CSV file
        file_path = r"C:\Users\Shivo\Desktop\RA BIZ TEK\Plastech\year_report.csv"
        df = pd.read_csv(file_path)
        data = df.to_dict('records')

        # Analysis prompt
        analysis_prompt = """Please analyze the manufacturing data and provide insights using these functions:

        1. Calculate machine performance using calculateAverageOEE with:
           - machine_column = "Machine Name"
           - oee_column = "OEE"

        2. Analyze downtime using calculateAverageUnplannedDowntime with:
           - machine_column = "Machine Name"
           - downtime_column = "Unplan D/T (Min.)"

        3. Predict future performance using predictOEEForNextMonth with:
           - oee_column = "OEE"
           - timestamp_column = "Timestamp"

        Please provide comprehensive insights from the analysis by making a table."""

        # Create a chat session
        chat = analyzer.model.start_chat()
        
        # Execute analysis with basic retry logic
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Create the message in the correct format
                message = {
                    "parts": [
                        {
                            "text": f"{analysis_prompt}\n\nData: {json.dumps(data)}"
                        }
                    ]
                }
                
                response = chat.send_message(message)
                message = {
                    "parts": [
                        {
                            "text": f"""
                            
                            Here's is the analysis workflow I am following: {analysis_prompt}. And this is my result:
{response}

convert this into a natural language answer that any process operator can easily understand and share the reasons behind your understanding as well. 

think step by step and follow my instructions.Data: {json.dumps(data)}"""
                        }
                    ]
                }
                break
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    logger.error(f"Failed to get response after {max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Attempt {retry_count} failed, retrying... Error: {str(e)}")
                time.sleep(2 ** retry_count)  # Exponential backoff
        
        # Process and store results
        results = analyzer.process_response(response)
        
        # Create results directory if it doesn't exist
        results_dir = 'analysis_results'
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Save results with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f'analysis_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Analysis completed successfully. Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        if not os.path.exists('error_logs'):
            os.makedirs('error_logs')
        error_file = os.path.join('error_logs', f'error_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(error_file, 'w') as f:
            f.write(f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error message: {str(e)}\n")
        logger.error(f"Error details saved to {error_file}")
    finally:
        logging.shutdown()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        logging.shutdown()
    
