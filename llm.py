import streamlit as st
from sparkai.embedding.spark_embedding import Embeddingmodel
from langchain_community.llms import SparkLLM
# Create the LLM
llm = SparkLLM(
        spark_api_url=st.secrets["SPARKAI_URL"],
        spark_app_id=st.secrets["SPARKAI_APP_ID"],
        spark_api_key=st.secrets["SPARKAI_API_KEY"],
        spark_api_secret=st.secrets["SPARKAI_API_SECRET"],
        spark_llm_domain=st.secrets["SPARKAI_DOMAIN"],
        streaming=True,
    )

