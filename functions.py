'''
[
  {
    "name": "calculateAverageGoodPartRate",
    "description": "Groups by machine and calculates the average good part rate to identify which machines are most efficient.",
    "parameters": {
      "type": "object",
      "properties": {
        "dataframe": {
          "type": "object",
          "description": "The input dataframe containing machine and rate data."
        },
        "machine_col": {
          "type": "string",
          "description": "The column name for machine names. Default is 'Machine Name'."
        },
        "rate_col": {
          "type": "string",
          "description": "The column name for good part rates. Default is 'Good_Part_Rate'."
        }
      }
    }
  },
  {
    "name": "calculateAverageUnplannedDowntime",
    "description": "Groups by machine and calculates the average unplanned downtime to identify machines with more frequent issues.",
    "parameters": {
      "type": "object",
      "properties": {
        "dataframe": {
          "type": "object",
          "description": "The input dataframe containing machine and downtime data."
        },
        "machine_col": {
          "type": "string",
          "description": "The column name for machine names. Default is 'Machine Name'."
        },
        "downtime_col": {
          "type": "string",
          "description": "The column name for unplanned downtime in minutes. Default is 'Unplan D/T (Min.)'."
        }
      }
    }
  },
  {
    "name": "calculateAverageGoodPartRateByMold",
    "description": "Groups by mold and calculates the average good part rate to see which molds produce higher quality parts.",
    "parameters": {
      "type": "object",
      "properties": {
        "dataframe": {
          "type": "object",
          "description": "The input dataframe containing mold and rate data."
        },
        "mold_col": {
          "type": "string",
          "description": "The column name for mold names. Default is 'Mold Name'."
        },
        "rate_col": {
          "type": "string",
          "description": "The column name for good part rates. Default is 'Good_Part_Rate'."
        }
      }
    }
  },
  {
    "name": "calculateAverageUnplannedDowntimeByMold",
    "description": "Groups by mold to see if certain molds are associated with more downtime.",
    "parameters": {
      "type": "object",
      "properties": {
        "dataframe": {
          "type": "object",
          "description": "The input dataframe containing mold and downtime data."
        },
        "mold_col": {
          "type": "string",
          "description": "The column name for mold names. Default is 'Mold Name'."
        },
        "downtime_col": {
          "type": "string",
          "description": "The column name for unplanned downtime in minutes. Default is 'Unplan D/T (Min.)'."
        }
      }
    }
  },
  {
    "name": "calculateDetailedPerformanceByMachineAndMold",
    "description": "Groups by both machine and mold to get a more detailed view of performance, including the average good part rate, the number of production runs, and the standard deviation (to see consistency).",
    "parameters": {
      "type": "object",
      "properties": {
        "dataframe": {
          "type": "object",
          "description": "The input dataframe containing machine, mold, and performance data."
        },
        "machine_col": {
          "type": "string",
          "description": "The column name for machine names. Default is 'Machine Name'."
        },
        "mold_col": {
          "type": "string",
          "description": "The column name for mold names. Default is 'Mold Name'."
        },
        "rate_col": {
          "type": "string",
          "description": "The column name for good part rates. Default is 'Good_Part_Rate'."
        }
      }
    }
  },
  {
    "name": "calculateAverageGoodPartRate",
    "description": "Groups data by operator and calculates the average Good Part Rate to identify the best-performing operators.",
    "parameters": {
      "type": "object",
      "properties": {
        "dataframe": {
          "type": "object",
          "description": "The input pandas DataFrame containing 'Operator' and 'Good_Part_Rate' columns."
        }
      },
      "required": [
        "dataframe"
      ]
    }
  },
  {
    "name": "analyzeOperatorPerformanceByMachine",
    "description": "Groups data by Operator and Machine Name to analyze performance, calculating the average Good Part Rate and the count of records for each combination.",
    "parameters": {
      "type": "object",
      "properties": {
        "dataframe": {
          "type": "object",
          "description": "The input pandas DataFrame containing 'Operator', 'Machine Name', and 'Good_Part_Rate' columns."
        }
      },
      "required": [
        "dataframe"
      ]
    }
  },
  {
    "name": "analyzeOperatorPerformanceByMold",
    "description": "Groups data by Operator and Mold Name to analyze performance, calculating the average Good Part Rate and the count of records for each combination.",
    "parameters": {
      "type": "object",
      "properties": {
        "dataframe": {
          "type": "object",
          "description": "The input pandas DataFrame containing 'Operator', 'Mold Name', and 'Good_Part_Rate' columns."
        }
      },
      "required": [
        "dataframe"
      ]
    }
  }
]
'''
