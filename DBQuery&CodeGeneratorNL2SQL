import os
from google.colab import userdata

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
print(os.getenv("OPENAI_API_KEY"))

print("OpenAI API key environment variable set. Remember to replace 'YOUR_API_KEY' with your actual key.")


import sqlite3

# Connect to the database
conn_schema = sqlite3.connect('my_database.db')
cursor = conn_schema.cursor()

# Query sqlite_master to get the schema of the 'customers' table
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='customers';")
db_schema = cursor.fetchone()[0]
conn_schema.close()

print("Database Schema for 'customers' table:\n", db_schema)

db_schema_tuple = cursor.fetchone()

if db_schema_tuple:
    db_schema = db_schema_tuple[0]
    print("Database Schema for 'customers' table:\n", db_schema)
else:
    db_schema = "Schema not found for 'customers' table. Please ensure the table is created."
    print(db_schema)

conn_schema.close()

import openai

# Assuming OPENAI_API_KEY is set in environment from a previous step

def generate_sql(query: str, db_schema: str) -> str:
    """Generates a SQL query from a natural language query and database schema using OpenAI."""

    prompt = f"""You are a highly skilled SQL query generator. Your task is to translate natural language questions into accurate SQL queries for a SQLite database.

Given the following database schema for the 'customers' table:
{db_schema}

Generate a SQLite SQL query that answers the following question:
'{query}'

Ensure the output is only the SQL query, without any additional text, explanations, or formatting like markdown code blocks. Also, do not add a semicolon at the end of the query.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # Or 'gpt-4' if available and preferred
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates SQL queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        sql_query = response.choices[0].message.content.strip()
        return sql_query
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""

# Test the function with a sample query and the obtained db_schema
sample_query = "Calculate the average annual income for each gender."
generated_sql_query = generate_sql(sample_query, db_schema)
print(f"Natural Language Query: {sample_query}")
print(f"Generated SQL Query: {generated_sql_query}")

import openai
import pandas as pd

def generate_python(query: str, df_head: str) -> str:
    """Generates Python code from a natural language query and DataFrame head using OpenAI."""

    prompt = f"""You are a highly skilled Python code generator. Your task is to translate natural language questions into accurate, runnable Python code. You will be provided with the head of a pandas DataFrame named `df` as context. The generated code should process the `df` DataFrame to answer the question.

Here is the head of the DataFrame `df` (in markdown format):
{df_head}

Generate Python code that answers the following question:
'{query}'

The code should assign its final result to a variable named `result`. Do not include any `import` statements for `pandas` or `numpy` as they are already available. Ensure the output is only the Python code, without any additional text, explanations, or formatting like markdown code blocks.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Or 'gpt-4' if available and preferred
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Python code."}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        python_code = response.choices[0].message.content.strip()
        return python_code
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""

# Test the function
sample_query_python = "Calculate the average annual income for each gender."
df_head_markdown = df.head().to_markdown(index=False)
generated_python_code = generate_python(sample_query_python, df_head_markdown)

print(f"Natural Language Query: {sample_query_python}")
print(f"Generated Python Code:\n{generated_python_code}")

import openai

def summarize(query: str, sql_result: dict, python_result: any) -> str:
    """Synthesizes information from SQL and Python execution results into a natural language summary using OpenAI."""

    prompt = f"""You are a helpful assistant tasked with summarizing information. You will be provided with an original natural language query, a result from a SQL query, and a result from a Python script.

Your goal is to synthesize all this information into a concise, natural language answer to the original query.

Original Query: '{query}'

SQL Result: {sql_result}

Python Result: {python_result}

Based on the above, provide a comprehensive natural language answer to the original query. The answer should be direct and without conversational filler or markdown code blocks. Start directly with the summary.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Or 'gpt-4' if available and preferred
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=300
        )
        summary_text = response.choices[0].message.content.strip()
        return summary_text
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return "An error occurred while generating the summary (OpenAI API Error)."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An error occurred while generating the summary."

# Test the function with sample data
sample_query_summary = "Which gender spends more on average and what is their average annual income?"
sql_result = pd.read_sql(generated_sql_query, conn_schema)

#dummy_python_result = {'gender': {'Female': 60, 'Male': 55}, 'annual_income': {'Female': 70000, 'Male': 65000}}
local_env = {"df": df}
exec(generated_python_code, local_env)
python_result = local_env["result"]

print(f"Original Query: {sample_query_summary}")
print(f"Sample SQL Result: {sql_result}")
print(f"Sample Python Result: {python_result}")
conn_schema.close()

generated_summary = summarize(sample_query_summary, sql_result, python_result)
print(f"\nGenerated Summary: {generated_summary}")


# Set Langchain tracing to v2
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Replace 'YOUR_LANGSMITH_API_KEY' with your actual Langsmith API key
# It is recommended to load this from a secure source in a production environment
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key if langsmith_api_key is not None else ""

# Optionally, set a project name for better organization in Langsmith
os.environ["LANGCHAIN_PROJECT"] = "LLM-powered-Chatbot-Mall-Customers"

print("Langsmith environment variables configured. Remember to replace 'YOUR_LANGSMITH_API_KEY' with your actual key if not already set as an environment variable.")
if langsmith_api_key is None or langsmith_api_key == "":
    print("WARNING: LANGSMITH_API_KEY environment variable is not set. Please set it for Langsmith tracing to work correctly.")
