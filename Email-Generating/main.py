from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.2,max_output_tokens=2000)

st.header("Generate Your Mail")

recipient=st.text_input("Recipient Email Address")
email_type=st.selectbox("Email Type",
    [
        "Job Application",
        "Formal",
        "Business",
        "Complaint",
        "Follow Up",
        "Thank You",
        "Meeting Request",
        "Leave Application"
    ]
)
tone=st.selectbox("Select Tone",[
    "Professional",
        "Friendly",
        "Formal",
        "Confident",
        "Persuasive"
])
purpose=st.text_area("Pursose of the Email")
extra_instruction=st.text_area("Extra Instruction")


template=load_prompt("template.json")



if st.button("Generate Email"):
    chain= template | model
    result=chain.invoke({
        "recipient":recipient,
        "email_type":email_type,
        "tone":tone,
        "purpose":purpose,
        "extra":extra_instruction

    })
    st.write(result.content)