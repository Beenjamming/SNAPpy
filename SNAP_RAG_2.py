
##################################################

import pandas as pd
import os
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from glob import glob
import json
import pandas as pd
import time
import shutil

api_key = os.environ.get('API_KEY')
azure_endpoint = 'xxx'
api_version = "2023-12-01-preview"

llm = AzureOpenAI(
    model="gpt-4-32k",
    deployment_name='gpt-4-32k',
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name='text-embedding-ada-002',
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from IPython.display import Markdown, display

import openai
from llama_index.output_parsers.guardrails import GuardrailsOutputParser
from llama_index.legacy.llm_predictor import StructuredLLMPredictor
openai.api_key = api_key

from llama_index.legacy.prompts import PromptTemplate
from llama_index.legacy.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_REFINE_PROMPT_TMPL,
)

from pydantic import BaseModel, Field
import guardrails as gd
from guardrails.validators import ValidRange, ValidChoices

class_desc = """
Classify the type of prescription from the note based on the options provided below:

SNAP (Safety Net Antibiotic Prescription) - Antibiotic prescribed for a bacterial infection that is only to be filled if the symptoms do not improve in 48-72 hours.
First Line Antibiotic
No Antibiotic Required
Antibiotic Prescribed for Another Reason
Other Reason for Antibiotic Prescription

"""

choices_list = ['SNAP','First Line Antibiotic','No Antibiotic Required','Antibiotic Prescribed For Another Reason','Other Reason for Antibiotic Prescription']

class Point(BaseModel):
    Classification: str = Field(description=class_desc,validators=[ValidChoices(choices=choices_list, on_fail='reask')])
    Medication: str = Field(description="Names of non antibiotic medications in the patient note.")
    Antibiotic: str = Field(description="Names of antibiotics in the patient note.")
    
df_results = pd.DataFrame(columns=['ID','PAT_ENC_CSN_ID','Classification','Antibiotic','Medication'])
excel_path = 'path'

df_notes = pd.read_excel(excel_path)
df_notes.describe()
#drop rows where Note1 is '' or NaN or 0
df_notes = df_notes[df_notes['Note1'] != '']
df_notes = df_notes[df_notes['Note1'].notnull()]
df_notes = df_notes[df_notes['Note1'] != 0]
df_notes.head()


df_notes.describe()


prompt = """
Query string here.

${gr.xml_prefix_prompt}

${output_schema}

${gr.json_suffix_prompt_v2_wo_none}
"""

query = '''
A SNAP (Safety Net Antibiotic Prescription) is a special prescription for children with acute otitis media that tells the parent to wait two days before they pick up the medication. Most acute otitis media gets better on its own and does not need an antibiotic. 
Often, pediatricians prescribe a SNAP so that parents will wait and see if their child improves on their own with "watchful waiting" before filling the prescription. 
Does this doctor's medical note say that:
- SNAP was prescribed
- regular first line antibiotics were prescribed
- No antibiotics were prescribed.
- If the patient was already on an antibiotic or is being prescribed an antibiotic for something other than Acute Otitis Media
- Other

'''

guard = gd.Guard.from_pydantic(output_class=Point, prompt=prompt)
output_parser = GuardrailsOutputParser(guard, llm=llm)

llm = AzureOpenAI(
    model="gpt-4-32k",
    deployment_name='gpt-4-32k',
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    output_parser=output_parser
)
llm_predictor = StructuredLLMPredictor(llm=llm)

alread_processed = pd.read_csv(r"C:\Temp\SNAP_RAG_o.csv")

df_notes = df_notes[~df_notes['PAT_ENC_CSN_ID'].isin(alread_processed['PAT_ENC_CSN_ID'])]
df_notes.describe()

#for row in df_notes.head(1).iterrows():
for row in df_notes.iterrows():
    print(row)
    try:       
        file = row[1]['Note1']
        #save the file as a txt file to be used in the VectorStoreIndex
        tmp_path = 'C:/Temp/note.txt'
        with open(tmp_path, 'w') as f:
            f.write(file)
        file = tmp_path
        file = [file]
        reader = SimpleDirectoryReader(input_files=file)
        document = reader.load_data()    
        index = VectorStoreIndex.from_documents(document, chunk_size=512,embed_model=embed_model)
        fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
        fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)
        qa_prompt = PromptTemplate(fmt_qa_tmpl, output_parser=output_parser)
        refine_prompt = PromptTemplate(fmt_refine_tmpl, output_parser=output_parser)
        query_engine = index.as_query_engine(
            text_qa_template=qa_prompt,
            refine_template=refine_prompt,
            llm_predictor=llm_predictor,
            )
        response = query_engine.query(query)
        print(response)
        structured_data = response.response
        print(structured_data)
        structured_data = json.loads(structured_data)
        df = pd.DataFrame([structured_data])
        df['PAT_ENC_CSN_ID'] = row[1]['PAT_ENC_CSN_ID']
        df['ID'] = row[1]['Random Number']
        #set columns to match the final dataframe
        df = df[['ID','PAT_ENC_CSN_ID','Classification','Antibiotic','Medication']]
        df_results = pd.concat([df_results,df])
        os.remove(tmp_path)
        #wait 3 seconds before processing the next file
        time.sleep(4)

    except Exception as e:
        print(f"Error with row {file}: {e}")


df_results.to_csv('C:/Temp/SNAP_RAG.csv',index=False)

