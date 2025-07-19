# main.py
# Backend for a collaborative dream-to-story AI generator.

import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- AI Model Initialization ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# --- AI Chains for Generation and Revision ---

# The Story Generation Chain
generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a master storyteller. A user will provide a dream description. Your task is to turn it into a compelling short story with a clear title. The story should be imaginative and well-structured. Respond with ONLY the title and the story, formatted like this:\n\nTITLE: [Your Title Here]\n\nSTORY: [Your story text here]"),
    ("human", "Dream Description: {dream_description}")
])
generation_chain = generation_prompt | llm | StrOutputParser()

# The Story Revision Chain
revision_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert editor. You will be given an existing story and a set of revision notes from the author. Your task is to rewrite the story, incorporating the feedback seamlessly while maintaining the original tone and core themes. Respond with ONLY the new title and the revised story, formatted like this:\n\nTITLE: [Your New Title Here]\n\nSTORY: [Your revised story text here]"),
    ("human", "Original Story:\n{original_story}\n\nAuthor's Feedback for Revision:\n{revision_notes}")
])
revision_chain = revision_prompt | llm | StrOutputParser()


# --- API Server Setup ---
app = FastAPI(title="DreamScript AI API")

# --- API Models ---
class DreamInput(BaseModel):
    dream_description: str

class RevisionInput(BaseModel):
    original_story: str
    revision_notes: str

class StoryOutput(BaseModel):
    title: str
    story_text: str

# --- Helper Function ---
def parse_story_output(llm_output: str) -> StoryOutput:
    """Helper function to parse the LLM's string output into a structured model."""
    try:
        title_part, story_part = llm_output.split("\n\nSTORY: ", 1)
        title = title_part.replace("TITLE: ", "").strip()
        story_text = story_part.strip()
        return StoryOutput(title=title, story_text=story_text)
    except ValueError:
        # Fallback if the model doesn't follow the format perfectly
        return StoryOutput(title="Untitled Story", story_text=llm_output)

# --- API Endpoints ---
@app.post("/generate-story", response_model=StoryOutput)
async def generate_story(dream_input: DreamInput):
    """Generates the initial story from a dream description."""
    llm_response = await generation_chain.ainvoke({
        "dream_description": dream_input.dream_description
    })
    return parse_story_output(llm_response)

@app.post("/revise-story", response_model=StoryOutput)
async def revise_story(revision_input: RevisionInput):
    """Revises an existing story based on user feedback."""
    llm_response = await revision_chain.ainvoke({
        "original_story": revision_input.original_story,
        "revision_notes": revision_input.revision_notes
    })
    return parse_story_output(llm_response)
