"""
Install an additional SDK for JSON schema support Google AI Python SDK

$ pip install google.ai.generativelanguage
"""

import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
  tools = [
    genai.protos.Tool(
      function_declarations = [
        genai.protos.FunctionDeclaration(
          name = "calculateAverageGoodPartRate",
          description = "Groups by machine and calculates the average good part rate to identify which machines are most efficient.",
          parameters = content.Schema(
            type = content.Type.OBJECT,
            properties = {
              "dataframe": content.Schema(
                type = content.Type.OBJECT,
                description = "The input dataframe containing machine and rate data.",
                properties = {
                },
              ),
              "machine_col": content.Schema(
                type = content.Type.STRING,
                description = "The column name for machine names. Default is 'Machine Name'.",
              ),
              "rate_col": content.Schema(
                type = content.Type.STRING,
                description = "The column name for good part rates. Default is 'Good_Part_Rate'.",
              ),
            },
          ),
        ),
        genai.protos.FunctionDeclaration(
          name = "calculateAverageUnplannedDowntime",
          description = "Groups by machine and calculates the average unplanned downtime to identify machines with more frequent issues.",
          parameters = content.Schema(
            type = content.Type.OBJECT,
            properties = {
              "dataframe": content.Schema(
                type = content.Type.OBJECT,
                description = "The input dataframe containing machine and downtime data.",
                properties = {
                },
              ),
              "machine_col": content.Schema(
                type = content.Type.STRING,
                description = "The column name for machine names. Default is 'Machine Name'.",
              ),
              "downtime_col": content.Schema(
                type = content.Type.STRING,
                description = "The column name for unplanned downtime in minutes. Default is 'Unplan D/T (Min.)'.",
              ),
            },
          ),
        ),
        genai.protos.FunctionDeclaration(
          name = "calculateAverageGoodPartRateByMold",
          description = "Groups by mold and calculates the average good part rate to see which molds produce higher quality parts.",
          parameters = content.Schema(
            type = content.Type.OBJECT,
            properties = {
              "dataframe": content.Schema(
                type = content.Type.OBJECT,
                description = "The input dataframe containing mold and rate data.",
                properties = {
                },
              ),
              "mold_col": content.Schema(
                type = content.Type.STRING,
                description = "The column name for mold names. Default is 'Mold Name'.",
              ),
              "rate_col": content.Schema(
                type = content.Type.STRING,
                description = "The column name for good part rates. Default is 'Good_Part_Rate'.",
              ),
            },
          ),
        ),
        genai.protos.FunctionDeclaration(
          name = "calculateAverageUnplannedDowntimeByMold",
          description = "Groups by mold to see if certain molds are associated with more downtime.",
          parameters = content.Schema(
            type = content.Type.OBJECT,
            properties = {
              "dataframe": content.Schema(
                type = content.Type.OBJECT,
                description = "The input dataframe containing mold and downtime data.",
                properties = {
                },
              ),
              "mold_col": content.Schema(
                type = content.Type.STRING,
                description = "The column name for mold names. Default is 'Mold Name'.",
              ),
              "downtime_col": content.Schema(
                type = content.Type.STRING,
                description = "The column name for unplanned downtime in minutes. Default is 'Unplan D/T (Min.)'.",
              ),
            },
          ),
        ),
        genai.protos.FunctionDeclaration(
          name = "calculateDetailedPerformanceByMachineAndMold",
          description = "Groups by both machine and mold to get a more detailed view of performance, including the average good part rate, the number of production runs, and the standard deviation (to see consistency).",
          parameters = content.Schema(
            type = content.Type.OBJECT,
            properties = {
              "dataframe": content.Schema(
                type = content.Type.OBJECT,
                description = "The input dataframe containing machine, mold, and performance data.",
                properties = {
                },
              ),
              "machine_col": content.Schema(
                type = content.Type.STRING,
                description = "The column name for machine names. Default is 'Machine Name'.",
              ),
              "mold_col": content.Schema(
                type = content.Type.STRING,
                description = "The column name for mold names. Default is 'Mold Name'.",
              ),
              "rate_col": content.Schema(
                type = content.Type.STRING,
                description = "The column name for good part rates. Default is 'Good_Part_Rate'.",
              ),
            },
          ),
        ),
        genai.protos.FunctionDeclaration(
          name = "calculateAverageGoodPartRate",
          description = "Groups data by operator and calculates the average Good Part Rate to identify the best-performing operators.",
          parameters = content.Schema(
            type = content.Type.OBJECT,
            enum = [],
            required = ["dataframe"],
            properties = {
              "dataframe": content.Schema(
                type = content.Type.OBJECT,
                description = "The input pandas DataFrame containing 'Operator' and 'Good_Part_Rate' columns.",
                properties = {
                },
              ),
            },
          ),
        ),
        genai.protos.FunctionDeclaration(
          name = "analyzeOperatorPerformanceByMachine",
          description = "Groups data by Operator and Machine Name to analyze performance, calculating the average Good Part Rate and the count of records for each combination.",
          parameters = content.Schema(
            type = content.Type.OBJECT,
            enum = [],
            required = ["dataframe"],
            properties = {
              "dataframe": content.Schema(
                type = content.Type.OBJECT,
                description = "The input pandas DataFrame containing 'Operator', 'Machine Name', and 'Good_Part_Rate' columns.",
                properties = {
                },
              ),
            },
          ),
        ),
        genai.protos.FunctionDeclaration(
          name = "analyzeOperatorPerformanceByMold",
          description = "Groups data by Operator and Mold Name to analyze performance, calculating the average Good Part Rate and the count of records for each combination.",
          parameters = content.Schema(
            type = content.Type.OBJECT,
            enum = [],
            required = ["dataframe"],
            properties = {
              "dataframe": content.Schema(
                type = content.Type.OBJECT,
                description = "The input pandas DataFrame containing 'Operator', 'Mold Name', and 'Good_Part_Rate' columns.",
                properties = {
                },
              ),
            },
          ),
        ),
      ],
    ),
  ],
  tool_config={'function_calling_config':'ANY'},
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")

# Print out each of the function calls requested from this single call.
# Note that the function calls are not executed. You need to manually execute the function calls.
# For more see: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Function_calling.ipynb
for part in response.parts:
  if fn := part.function_call:
    args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
    print(f"{fn.name}({args})")
