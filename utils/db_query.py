import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

def load_and_display_database():
    try:
        df = pd.read_csv("database.csv")

        df['Start Time'] = df['Start Time'].apply(lambda x: datetime.fromtimestamp(float(x)).strftime('%d/%m/%Y %H:%M:%S'))
        df['End Time'] = df['End Time'].apply(lambda x: datetime.fromtimestamp(float(x)).strftime('%d/%m/%Y %H:%M:%S'))

        st.dataframe(df)  
        return df
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return None


def query_database(query_type, person_name=None, query_date=None, location=None):
    df = pd.read_csv("database.csv")

    df['Start Time'] = df['Start Time'].apply(lambda x: datetime.fromtimestamp(float(x)).strftime('%d/%m/%Y %H:%M:%S'))
    df['End Time'] = df['End Time'].apply(lambda x: datetime.fromtimestamp(float(x)).strftime('%d/%m/%Y %H:%M:%S'))

    if query_type == "date_query" and query_date:
        query_date_str = query_date.strftime('%d/%m/%Y')  
        filtered_df = df[df['Start Time'].str.startswith(query_date_str)]
        
        if not filtered_df.empty:
            result = filtered_df[['Person Name', 'Start Time', 'End Time', 'Location']]
            return result
        else:
            return f"No results found for the date: {query_date_str}"

    elif query_type == "person_location_query" and person_name:
        if location:
            filtered_df = df[(df['Person Name'].str.contains(person_name, case=False)) & (df['Location'].str.contains(location, case=False))]
        else:
            filtered_df = df[df['Person Name'].str.contains(person_name, case=False)]

        if not filtered_df.empty:
            result = filtered_df[['Person Name', 'Start Time', 'End Time', 'Location']]
            return result
        else:
            return f"No results found for {person_name}" + (f" at {location}" if location else "")
    
    else:
        return "Invalid query parameters."
    

def generate_person_path(person_name, query_date):
    df = pd.read_csv("database.csv")

    df['Start Time'] = df['Start Time'].apply(lambda x: datetime.fromtimestamp(float(x)).strftime('%d/%m/%Y %H:%M:%S'))
    df['End Time'] = df['End Time'].apply(lambda x: datetime.fromtimestamp(float(x)).strftime('%d/%m/%Y %H:%M:%S'))

    query_date_str = query_date.strftime('%d/%m/%Y')  
    filtered_df = df[(df['Start Time'].str.startswith(query_date_str)) & (df['Person Name'].str.contains(person_name, case=False))]

    if not filtered_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_df['Start Time'], filtered_df['Location'], marker='o', linestyle='-', color='blue', label=person_name)

        plt.title(f"Movement Path for {person_name} on {query_date_str}")
        plt.xlabel("Time")
        plt.ylabel("Location")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        st.pyplot(plt)
    else:
        st.warning(f"No data found for {person_name} on {query_date_str}")

