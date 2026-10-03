import dspy
import pandas as pd

#to get history
def get_history(lm, n):
    history = lm.history
    last_history = {}
    if len(history) >= n:
        last_history['system'] = history[-n:][0]['messages'][0]['content']
        last_history['user'] = history[-n:][0]['messages'][1]['content']
    return last_history

def process_df(df, input_columns, output_column):
    """Select the configured columns and lowercase every value.

    Lowercasing matters because the evaluation metric is `answer_exact_match`:
    'Mars' and 'mars' would otherwise be scored as different answers, so a
    correct prediction could be marked wrong purely on casing.

    (The previous version accepted `input_columns` / `output_column` and then
    ignored both, lowercasing the whole frame regardless.)
    """
    columns = list(input_columns) + list(output_column)
    processed_df = df[columns].copy()
    return processed_df.map(lambda x: str(x).lower() if x is not None else x)