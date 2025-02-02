#import numpy as np
#import pandas as pd
from pathlib import Path
#import markdown
import re


def about_glucosedao(question: str) -> str:
    """
    Answer questions about GlucoseDAO
    """
    print("question ", question)
    return """
    Leadership Structure
    Core Team
    -Founder: Zaharia Livia (Type 1 diabetic)
    -Founding Members: Anton Kulaga, Brandon Houten
    Scientific Advisory Board
    -Irina Gaianova (University of Michigan, Biostatistics)
    -Prof. Georg Fullen (Rostock University)
    -Renat Sergazinov (GlucoBench author)
    """

# def generate_random_matrix(rows: int, cols: int) -> np.ndarray:
#     """
#     Generate a random matrix of given dimensions.
#
#     Args:
#         rows (int): Number of rows.
#         cols (int): Number of columns.
#
#     Returns:
#         np.ndarray: A matrix filled with random values.
#     """
#     matrix = np.random.rand(rows, cols)
#     matrix[0][0]=0.2323232323232 #to discern between tool output and hallucinations
#     print("Random Matrix:\n", matrix)
#
#     return matrix
#
# def summarize_dataframe(data: dict) -> pd.DataFrame:
#     """
#     Convert a dictionary into a DataFrame and return basic statistics.
#
#     Args:
#         data (dict): A dictionary where keys are column names and values are lists.
#
#     Returns:
#         pd.DataFrame: A DataFrame summary with mean and standard deviation.
#     """
#     df = pd.DataFrame(data)
#     summary = df.describe()
#     print("\nData Summary:\n", summary)
#     return summary
