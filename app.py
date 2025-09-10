import streamlit as st
from langchain.llms import Bedrock
from langchain.prompts import PromptTemplate
import boto3

# --- Streamlit UI ---
st.set_page_config(page_title="AWS Bedrock + LangChain Mini App", page_icon="🤖")

st.title("🤖 Mini Bedrock + LangChain App")
st.write("Type your prompt and get a concise answer (max 100 words).")

# Input box
user_input = st.text_area("Enter your prompt:")

# Button
if st.button("Ask AI"):
    if not user_input.strip():
        st.warning("Please enter a prompt.")
    else:
        # --- Bedrock client setup ---
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"  # Change region if needed
        )

        # LangChain Bedrock LLM
        llm = Bedrock(
            client=bedrock_client,
            model_id="amazon.titan-text-lite-v1"  # Or "amazon.titan-text-lite-v1"
        )

        # Prompt template
        template = """You are a helpful assistant.
        Answer the following question in less than 100 words:
        {question}"""

        prompt = PromptTemplate.from_template(template).format(question=user_input)

        # Get response
        response = llm(prompt)

        # Show reply
        st.subheader("Reply:")
        st.write(response.strip())
