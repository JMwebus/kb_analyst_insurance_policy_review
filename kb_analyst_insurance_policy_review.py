# Streamlit Two Column Insurance Policy Analyst



# llama-parse is async-first, running the sync code in a notebook requires the use of nest_asyncio

# python 3.113


import os
import yaml
import streamlit as st
import nest_asyncio


from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

# Create code for os.environ
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-..."
os.environ["OPENAI_API_KEY"] = "sk-..."

# Load all keys from the credentials file
credentials = yaml.safe_load(open("../credentials/credentials.yml"))

# Set environment variables for both APIs if necessary
os.environ["LLAMA_CLOUD_API_KEY"] = credentials["LLAMA_CLOUD_API_KEY"]
os.environ["OPENAI_API_KEY"] = credentials["OPENAI_API_KEY"]

# Enable asynchronous support
nest_asyncio.apply()

# for the purpose of this example, we will use the small model embedding and gpt3.5
embed_model = OpenAIEmbedding(model="text-embedding-3-small")                           # Validate if the embedding model is available
llm = OpenAI(model="gpt-4O", temperature=0.7)

Settings.llm = llm

from llama_parse import LlamaParse

# Get the insurance policy document
# !wget "https://policyholder.gov.in/documents/37343/931203/NBHTGBP22011V012223.pdf/c392bcc1-f6a8-cadd-ab84-495b3273d2c3?version=1.0&t=1669350459879&download=true" -O "./policy.pdf" 

# VAINILLA PARSING ??
# Parse the policy
documents = LlamaParse(result_type="markdown").load_data("./policy.pdf")



from llama_index.core.node_parser import MarkdownElementNodeParser

node_parser = MarkdownElementNodeParser(
    llm=OpenAI(model=llm), num_workers=8
)

nodes = node_parser.get_nodes_from_documents(documents)

base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

recursive_index = VectorStoreIndex(nodes=base_nodes + objects)

query_engine = recursive_index.as_query_engine(similarity_top_k=25)




documents_with_instruction = LlamaParse(
    result_type="markdown",
    parsing_instruction="""
This document is an insurance policy.
When a benefits/coverage/exlusion is describe in the document ammend to it add a text in the follwing benefits string format (where coverage could be an exclusion).

For {nameofrisk} and in this condition {whenDoesThecoverageApply} the coverage is {coverageDescription}. 
                                        
If the document contain a benefits TABLE that describe coverage amounts, do not ouput it as a table, but instead as a list of benefits string.
                                       
""",
).load_data("./policy.pdf")


node_parser_instruction = MarkdownElementNodeParser(
    llm=OpenAI(model="gpt-3.5-turbo-0125"), num_workers=8
)
nodes_instruction = node_parser.get_nodes_from_documents(documents_with_instruction)
(
    base_nodes_instruction,
    objects_instruction,
) = node_parser_instruction.get_nodes_and_objects(nodes_instruction)

recursive_index_instruction = VectorStoreIndex(
    nodes=base_nodes_instruction + objects_instruction
)
query_engine_instruction = recursive_index_instruction.as_query_engine(
    similarity_top_k=25
)