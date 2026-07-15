from langchain_core.prompts import PromptTemplate

template=PromptTemplate(
    template="""You are an expert email writer.

Write a complete email.

Recipient:
{recipient}

Email Type:
{email_type}

Tone:
{tone}

Purpose:
{purpose}

Additional Instructions:
{extra}

Return:

Subject

Greeting

Body

Closing"""
)


template.save("template.json")