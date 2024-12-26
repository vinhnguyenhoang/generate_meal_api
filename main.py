from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
app = FastAPI()

def load_data_from_csv(file_path):
    """Load data from a CSV file and return a list of Document objects."""
    df = pd.read_csv(file_path)
    documents = [
        Document(page_content=f"Meal_Id: {row['meal_id']}\nMeal_Name: {row['meal_name']}\nIngredient_Name: {row['ingredient_name']}\nIngredient_Quantity: ({row['ingredient_qty']})")
        for _, row in df.iterrows()
    ]
    return documents, df

def create_retrieval_chain(documents, model_name, groq_api_key):
    """Create a RetrievalQA chain using the given documents and model."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    model = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever)

    return qa_chain

def generate_question(preferences):
    """Generate a question string based on user preferences."""
    question = f"""
Given the following user preferences:
- Height: {preferences['height']} cm
- Weight: {preferences['weight']} kg
- Age: {preferences['age']} years
- Preferences: {preferences['preferences']}
- Lifestyle: {preferences['lifestyle']}
- Diet: {preferences['diet']}
- Allergies: {preferences['allergy']}
- Previous meals: {', '.join(preferences['previous_meals'])}

Suggest a suitable meal for user from the following list of meals and ingredients.
Do not include any explanation or additional text. Return only the array of meal_ids, about 5 to 10 suitable meal_id.
"""
    return question

class Preferences(BaseModel):
    height: int
    weight: int
    age: int
    preferences: str
    lifestyle: str
    diet: str
    allergy: str
    previous_meals: List[str]

@app.post("/recommend")
async def recommend_meals(preferences: Preferences, file_path: str = "meal_ingredients.csv"):
    try:
        documents, df = load_data_from_csv(file_path)

        # Create RetrievalQA chain
        groq_api_key = api_key
        model_name = "all-MiniLM-L6-v2"
        qa_chain = create_retrieval_chain(documents, model_name, groq_api_key)

        # Generate question
        question = generate_question(preferences.dict())

        # Query the model
        result = qa_chain.invoke(question)

        return {"recommended_meal_ids": result['result']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Meal Recommendation API!"}
# Start the server by using uvicorn
# uvicorn main:app --reload
