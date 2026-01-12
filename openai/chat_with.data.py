import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI


def oppsummer_med_ai():
    # --- Setup ---
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # --- Load data ---
    url = "https://raw.githubusercontent.com/jensmorten/onesixtynine/main/data/pollofpolls_master.csv"
    df = pd.read_csv(url, index_col="Mnd", parse_dates=True)
    df = df.sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")

    # --- Prepare data for the model ---
    data_description = f"""
    Du skal analysere meiningsmålingsdata i Noreg

    Kolonner (Parti):
    {list(df.columns)}

    antall rader: {len(df)}

    Dato :
    {df.index.min()} to {df.index.max()}

    Sample rows:
    {df.head(5).to_csv()}
    """

    # --- Ask a question ---
    question = "Oppsummer trendene i datasettet. Fokuser på tid etter valget i oktober 2025, korrelasjonar mellom partia og prediker mulig utvikling opptil eitt år fram i tid."

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": "Du er ein analytikar som helper med å tolke datasettet. Skriv på korrekt nynorsk"
            },
            {
                "role": "user",
                "content": data_description
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.output_text


def still_eige_spm(q):  
    # --- Setup ---
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # --- Load data ---
    url = "https://raw.githubusercontent.com/jensmorten/onesixtynine/main/data/pollofpolls_master.csv"
    df = pd.read_csv(url, index_col="Mnd", parse_dates=True)
    df = df.sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")

    # --- Prepare data for the model ---
    data_description = f"""
    Du skal analysere meiningsmålingsdata i Noreg

    Kolonner (Parti):
    {list(df.columns)}

    antall rader: {len(df)}

    Dato :
    {df.index.min()} to {df.index.max()}

    Sample rows:
    {df.head(5).to_csv()}
    """

    # --- Ask a question ---

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": "Du er ein analytikar som helper med å tolke datasettet. Skriv på korrekt nynorsk"
            },
            {
                "role": "user",
                "content": data_description
            },
            {
                "role": "user",
                "content": q
            }
        ]
    )

    return response.output_text