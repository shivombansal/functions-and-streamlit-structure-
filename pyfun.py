import pandas as pd

def calculate_average_good_part_rate(dataframe, machine_col='Machine Name', rate_col='Good_Part_Rate'):
    """
    Groups by machine and calculates the average good part rate to identify which machines are most efficient.
    """
    return dataframe.groupby(machine_col)[rate_col].mean().reset_index()

def calculate_average_unplanned_downtime(dataframe, machine_col='Machine Name', downtime_col='Unplan D/T (Min.)'):
    """
    Groups by machine and calculates the average unplanned downtime to identify machines with more frequent issues.
    """
    return dataframe.groupby(machine_col)[downtime_col].mean().reset_index()

def calculate_average_good_part_rate_by_mold(dataframe, mold_col='Mold Name', rate_col='Good_Part_Rate'):
    """
    Groups by mold and calculates the average good part rate to see which molds produce higher quality parts.
    """
    return dataframe.groupby(mold_col)[rate_col].mean().reset_index()

def calculate_average_unplanned_downtime_by_mold(dataframe, mold_col='Mold Name', downtime_col='Unplan D/T (Min.)'):
    """
    Groups by mold to see if certain molds are associated with more downtime.
    """
    return dataframe.groupby(mold_col)[downtime_col].mean().reset_index()

def calculate_detailed_performance_by_machine_and_mold(dataframe, machine_col='Machine Name', mold_col='Mold Name', rate_col='Good_Part_Rate'):
    """
    Groups by both machine and mold to get a more detailed view of performance, including the average good part rate, the number of production runs, and the standard deviation (to see consistency).
    """
    return dataframe.groupby([machine_col, mold_col])[rate_col].agg(['mean', 'count', 'std']).reset_index()

def calculate_average_good_part_rate_by_operator(dataframe):
    """
    Groups data by operator and calculates the average Good Part Rate to identify the best-performing operators.
    """
    return dataframe.groupby('Operator')['Good_Part_Rate'].mean().reset_index()

def analyze_operator_performance_by_machine(dataframe):
    """
    Groups data by Operator and Machine Name to analyze performance, calculating the average Good Part Rate and the count of records for each combination.
    """
    return dataframe.groupby(['Operator', 'Machine Name'])['Good_Part_Rate'].agg(['mean', 'count']).reset_index()

def analyze_operator_performance_by_mold(dataframe):
    """
    Groups data by Operator and Mold Name to analyze performance, calculating the average Good Part Rate and the count of records for each combination.
    """
    return dataframe.groupby(['Operator', 'Mold Name'])['Good_Part_Rate'].agg(['mean', 'count']).reset_index()
