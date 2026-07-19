from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from pypdf import PdfReader
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

# model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")

st.set_page_config(
    page_title="AI RESUME ANALYZER",
     page_icon="📄",
    layout="wide"
)


class Person(BaseModel):
    name:str=Field(description="Name of the person")
    education:list[str]=Field(description="Education of the person acadamic education ")
    skills:list[str]=Field(description="Technical Skills of the person")
    experience:Optional[list[str]]=Field(default=None,description="Experience of the person")
    projects:Optional[list[str]]=Field(default=None,description="Projects of the person")
    strengths:list[str]=Field(description="Strenght of the person")
    weaknesses:list[str]=Field(description="weakness of the person")
    resume_score: int=Field(ge=0,le=10,description="Give Score to resume btween 0 to 10")
    suggested_jobs: list[str]=Field(description="Suggest job according to resume")

parser=PydanticOutputParser(pydantic_object=Person)

prompt = PromptTemplate(
    template="""
You are an experienced HR Manager.

Analyze the following resume carefully.

Resume:
{resume}

Extract:

- Name
- Education(study)
- Skills
- Experience
- Projects
- Strengths
- Weaknesses
- Resume Score
- Suggested Jobs

{format_instruction}""",
 input_variables=["resume"],
    partial_variables={"format_instruction": parser.get_format_instructions()   }
)



st.title("AI Resume Analyzer")

st.write("Upload your Resume")

uploaded_file=st.file_uploader(
    "Upload Resume(pdf only)",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success("Resume Uploaded Successfully!")

    reader = PdfReader(uploaded_file)

    resume_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            resume_text += text + "\n"

    st.subheader("Extracted Resume Text")
    chain=prompt | model  | parser

    result=chain.invoke({"resume":resume_text})
    st.header("Resume Analysis")

    st.write("### Name")
    st.write(result.name)

    st.write("### Education")
    st.write(result.education)

    st.write("### Skills")
    st.write(result.skills)

    st.write("### Experience")
    st.write(result.experience)

    st.write("### Projects")
    st.write(result.projects)

    st.write("### Strengths")
    st.write(result.strengths)

    st.write("### Weaknesses")
    st.write(result.weaknesses)

    st.write("### Resume Score")
    st.write(result.resume_score)

    st.write("### Suggested Jobs")
    st.write(result.suggested_jobs)

  
    
       

    

    





