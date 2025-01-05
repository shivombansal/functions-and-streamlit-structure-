from dotenv import load_dotenv
import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

# Create the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Define common dataframe properties
df_properties = {
    "Machine Name": content.Schema(
        type=content.Type.STRING,
        description="Name of the manufacturing machine"
    ),
    "Mold Name": content.Schema(
        type=content.Type.STRING,
        description="Name of the mold being used"
    ),
    "Operator": content.Schema(
        type=content.Type.STRING,
        description="Name of the operator"
    ),
    "Good_Part_Rate": content.Schema(
        type=content.Type.NUMBER,
        description="Rate of good parts produced"
    ),
    "Unplan D/T (Min.)": content.Schema(
        type=content.Type.NUMBER,
        description="Unplanned downtime in minutes"
    ),
    "Timestamp": content.Schema(
        type=content.Type.STRING,
        description="Time of the record"
    )
}

tools = [
    genai.protos.Tool(
        function_declarations=[
            # 1. Average Good Part Rate by Machine
            genai.protos.FunctionDeclaration(
                name="calculateAverageGoodPartRate",
                description="Calculates the average good part rate grouped by machine.",
                parameters=content.Schema(
                    type=content.Type.OBJECT,
                    properties={
                        "dataframe": content.Schema(
                            type=content.Type.OBJECT,
                            description="The input dataframe with machine and rate data.",
                            properties=df_properties
                        ),
                        "machine_col": content.Schema(
                            type=content.Type.STRING,
                            description="The column name for machine names."
                        ),
                        "rate_col": content.Schema(
                            type=content.Type.STRING,
                            description="The column name for good part rates."
                        ),
                    },
                ),
            ),
            # 2. Average Unplanned Downtime by Machine
            genai.protos.FunctionDeclaration(
                name="calculateAverageUnplannedDowntime",
                description="Groups by machine and calculates average unplanned downtime.",
                parameters=content.Schema(
                    type=content.Type.OBJECT,
                    properties={
                        "dataframe": content.Schema(
                            type=content.Type.OBJECT,
                            description="The input dataframe with machine and downtime data.",
                            properties=df_properties
                        ),
                        "machine_col": content.Schema(
                            type=content.Type.STRING,
                            description="The column name for machine names."
                        ),
                        "downtime_col": content.Schema(
                            type=content.Type.STRING,
                            description="The column name for unplanned downtime."
                        ),
                    },
                ),
            ),
            # 3. Average Good Part Rate by Mold
            genai.protos.FunctionDeclaration(
                name="calculateAverageGoodPartRateByMold",
                description="Groups by mold and calculates average good part rate.",
                parameters=content.Schema(
                    type=content.Type.OBJECT,
                    properties={
                        "dataframe": content.Schema(
                            type=content.Type.OBJECT,
                            description="The input dataframe with mold and rate data.",
                            properties=df_properties
                        ),
                        "mold_col": content.Schema(
                            type=content.Type.STRING,
                            description="The column name for mold names."
                        ),
                        "rate_col": content.Schema(
                            type=content.Type.STRING,
                            description="The column name for good part rates."
                        ),
                    },
                ),
            ),
            # 4. Average Unplanned Downtime by Mold
            genai.protos.FunctionDeclaration(
                name="calculateAverageUnplannedDowntimeByMold",
                description="Groups by mold to analyze downtime patterns.",
                parameters=content.Schema(
                    type=content.Type.OBJECT,
                    properties={
                        "dataframe": content.Schema(
                            type=content.Type.OBJECT,
                            description="The input dataframe with mold and downtime data.",
                            properties=df_properties
                        ),
                        "mold_col": content.Schema(
                            type=content.Type.STRING,
                            description="The column name for mold names."
                        ),
                        "downtime_col": content.Schema(
                            type=content.Type.STRING,
                            description="The column name for unplanned downtime."
                        ),
                    },
                ),
            ),
            # 5. Detailed Performance by Machine and Mold
            genai.protos.FunctionDeclaration(
                name="calculateDetailedPerformanceByMachineAndMold",
                description="Detailed performance analysis by machine and mold combination.",
                parameters=content.Schema(
                    type=content.Type.OBJECT,
                    properties={
                        "dataframe": content.Schema(
                            type=content.Type.OBJECT,
                            description="The input dataframe with performance data.",
                            properties=df_properties
                        ),
                        "machine_col": content.Schema(
                            type=content.Type.STRING,
                            description="The column name for machine names."
                        ),
                        "mold_col": content.Schema(
                            type=content.Type.STRING,
                            description="The column name for mold names."
                        ),
                        "rate_col": content.Schema(
                            type=content.Type.STRING,
                            description="The column name for good part rates."
                        ),
                    },
                ),
            ),
            # 6. Average Good Part Rate by Operator
            genai.protos.FunctionDeclaration(
                name="calculateAverageGoodPartRatebyoperator",
                description="Analyzes operator performance through good part rates.",
                parameters=content.Schema(
                    type=content.Type.OBJECT,
                    properties={
                        "dataframe": content.Schema(
                            type=content.Type.OBJECT,
                            description="The input dataframe with operator performance data.",
                            properties=df_properties
                        ),
                    },
                ),
            ),
            # 7. Operator Performance by Machine
            genai.protos.FunctionDeclaration(
                name="analyzeOperatorPerformanceByMachine",
                description="Analyzes operator performance on different machines.",
                parameters=content.Schema(
                    type=content.Type.OBJECT,
                    properties={
                        "dataframe": content.Schema(
                            type=content.Type.OBJECT,
                            description="The input dataframe with operator and machine data.",
                            properties=df_properties
                        ),
                    },
                ),
            ),
            # 8. Operator Performance by Mold
            genai.protos.FunctionDeclaration(
                name="analyzeOperatorPerformanceByMold",
                description="Analyzes operator performance with different molds.",
                parameters=content.Schema(
                    type=content.Type.OBJECT,
                    properties={
                        "dataframe": content.Schema(
                            type=content.Type.OBJECT,
                            description="The input dataframe with operator and mold data.",
                            properties=df_properties
                        ),
                    },
                ),
            ),
        ],
    )
]

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    tools=tools,
    tool_config={'function_calling_config': 'ANY'},
)

# File path (using raw string to handle Windows path)
file_path = r"path to file"
uploaded_file = upload_to_gemini(file_path, mime_type="text/csv")

# Start chat session
chat_session = model.start_chat(
    history=[
        {
            "role": "model",
            "parts": [
                uploaded_file.uri
            ],
        },
    ]
)

# Send a message and process response
response = chat_session.send_message("Please analyze the manufacturing data and provide insights about machine and operator performance.")

# Process the response
for part in response.parts:
    if fn := part.function_call:
        args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
        print(f"{fn.name}({args})")
    else:
        print(part.text)
        
