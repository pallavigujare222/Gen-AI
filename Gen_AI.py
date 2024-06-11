import streamlit as st
import pandas as pd
from langchain.document_loaders import PDFLoader, CSVLoader, ExcelLoader
from langchain.text_splitter import SentenceSplitter
from langchain.llms import OpenAI
from langchain.chains import EntityExtractionChain
from langchain.retrievers import RAGRetriever
from openai import Completion

# Initialize OpenAI API
openai_api_key = "YOUR_OPENAI_API_KEY"

# Load financial statements
def load_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        loader = PDFLoader(file_path=uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        loader = CSVLoader(file_path=uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        loader = ExcelLoader(file_path=uploaded_file)
    return loader.load()

# Extract entities
def extract_entities(documents):
    splitter = SentenceSplitter(chunk_size=512)
    chunks = splitter.split_documents(documents)
    
    llm = OpenAI(api_key=openai_api_key)
    entity_extraction_chain = EntityExtractionChain(llm=llm, entity_types=["Revenue", "Net Income", "Assets", "Liabilities"])
    return entity_extraction_chain.run(chunks)

# Generate detailed analysis
def generate_analysis_report(entities):
    analysis_prompt = f"Analyze the following financial data and provide a detailed report:\n{entities}"
    response = Completion.create(
        engine="davinci",
        prompt=analysis_prompt,
        max_tokens=1000
    )
    return response.choices[0].text

# Identify compliance issues
def identify_compliance_issues(entities, regulations_documents):
    llm = OpenAI(api_key=openai_api_key)
    rag_retriever = RAGRetriever(documents=regulations_documents, llm=llm)
    relevant_regulations = rag_retriever.retrieve(query="compliance issues for the provided financial data")
    
    compliance_prompt = f"Identify compliance issues in the following financial data based on these regulations:\n{entities}\n\nRegulations:\n{relevant_regulations}"
    response = Completion.create(
        engine="davinci",
        prompt=compliance_prompt,
        max_tokens=500
    )
    return response.choices[0].text

# Streamlit app
def main():
    st.title("Financial Statement Analyzer")

    # File upload
    uploaded_file = st.file_uploader("Upload your financial statement", type=["pdf", "csv", "xlsx"])

    if uploaded_file is not None:
        # Load and process the uploaded file
        documents = load_file(uploaded_file)
        entities = extract_entities(documents)
        
        # Generate and display analysis report
        analysis_report = generate_analysis_report(entities)
        st.header("Analysis Report")
        st.write(analysis_report)
        
        # Load regulatory documents (for simplicity, using a placeholder function)
        regulations_documents = load_file("regulations.pdf")  # Replace with actual path to regulations file
        
        # Generate and display compliance report
        compliance_report = identify_compliance_issues(entities, regulations_documents)
        st.header("Compliance Report")
        st.write(compliance_report)

if __name__ == "__main__":
    main()
