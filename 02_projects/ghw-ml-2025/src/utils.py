import pandas as pd

def load_standard_model():
    df = pd.read_csv("data/Student_Performance_Dataset.csv");
    allowed_columns = [
        "Attendance",
        "Midterm_Score",
        "Final_Score",
        "Projects_Score",
        "Study_Hours"]
    
    return df[allowed_columns]

def load_biased_model():
    df = pd.read_csv("data/Student_Performance_Dataset.csv");
    biased_allowed_columns = [
        "Attendance",
        "Midterm_Score",
        "Final_Score",
        "Projects_Score",
        "Study_Hours",
        "Gender",
        "Parent_Education_Level",
        "Internet_Access_at_Home"
    ]
    
    return df[biased_allowed_columns]