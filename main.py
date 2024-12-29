from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import re
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
app = FastAPI()

def load_data_from_csv(file_path):
    """Load data from a CSV file and return a list of Document objects."""
    df = pd.read_csv(file_path)
    documents = [
        Document(
            page_content=f"Meal_Id: {row['meal_id']}\nMeal_Name: {row['meal_name']}\nIngredient_Name: {row['ingredient_name']}\nIngredient_Quantity: ({row['ingredient_qty']}\nCalo: ({row['calo']})")
        for _, row in df.iterrows()
    ]
    return documents, df
def create_custom_retriever(vectorstore, preferences):
    """Create a custom retriever that filters results based on user preferences."""
    def custom_filter(doc):
        content = doc.page_content.lower()
        if preferences["allergy"].lower() in content:
            return False  # Exclude documents containing allergens
        return True
    
    return vectorstore.as_retriever(filter_function=custom_filter)
def create_retrieval_chain(documents, model_name, groq_api_key, preferences):
    """Create a RetrievalQA chain using the given documents and model."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = retriever = create_custom_retriever(vectorstore, preferences)


    model = ChatGroq(model=model_name, api_key=groq_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever)

    return qa_chain

def generate_question(preferences):
    """Generate a structured prompt using PromptTemplate."""
    template = """
You are a highly accurate and concise recommendation engine. Based on the user's preferences below, suggest the most suitable meals:
- Height: {height} cm
- Weight: {weight} kg
- Age: {age} years
- Total daily calories: {calo}
- Preferences: {preferences}
- Lifestyle: {lifestyle}
- Diet: {diet}
- Allergies: {allergy}
- Previous meals: {previous_meals}

Based on the provided list of meals and ingredients, please recommend unique combinations of meal IDs. Ensure the following:
- Each combination contains exactly 3 unique meal IDs.
- No duplicate or repeated meal IDs within or across combinations.
- The total calories of each combination is within the range ({calo} - 100) to ({calo} + 100).

Return a list containing groups of 3 unique meal IDs, with each group representing a combination of 3 meals.
Do not include any explanation or additional text in the response.
"""
    prompt = PromptTemplate(
        input_variables=[
            "height", "weight", "age", "calo", "preferences",
            "lifestyle", "diet", "allergy", "previous_meals"
        ],
        template=template
    )
    return prompt.format(
        height=preferences["height"],
        weight=preferences["weight"],
        age=preferences["age"],
        calo=preferences["calo"],
        preferences=preferences["preferences"],
        lifestyle=preferences["lifestyle"],
        diet=preferences["diet"],
        allergy=preferences["allergy"],
        previous_meals=", ".join(preferences["previous_meals"]),
    )

class Preferences(BaseModel):
    height: Optional[int] = None
    weight: Optional[int] = None
    age: Optional[int] = None
    calo: Optional[int] = None
    preferences: str = ""
    lifestyle: str = ""
    diet: str = ""
    allergy: str = ""
    previous_meals: List[str] = []

def format_output(raw_output, df):
    """Extract a list of unique integer meal IDs from the model's raw output."""
    try:
        # Find array 
        matches = re.findall(r"\[([^\]]+)\]", raw_output)
        if not matches:
            raise ValueError("No array found in the output.")
        
        array_content = matches[0]
        
        meal_ids = [int(item.strip()) for item in array_content.split(",") if item.strip().isdigit()]
        # check id in data frame
        valid_meal_ids = [meal_id for meal_id in set(meal_ids) 
                               if meal_id in df['meal_id'].values]
        
        if not valid_meal_ids:
            raise ValueError("No new meal IDs to recommend. All results are duplicates or invalid.")
        
        
        return valid_meal_ids
    except Exception as e:
        raise ValueError(f"Error parsing output: {str(e)}")

@app.post("/recommend")
async def recommend_meals(preferences: Preferences, file_path: str = "meal_ingredients.csv"):
    try:
        documents, df = load_data_from_csv(file_path)

        # Create RetrievalQA chain
        groq_api_key = api_key
        model_name = "llama-3.3-70b-versatile"
        qa_chain = create_retrieval_chain(documents, model_name, groq_api_key, preferences)

        # Generate question
        question = generate_question(preferences.model_dump())
        
        # Query the model
        raw_result = qa_chain.invoke(question)
        result = format_output(raw_result["result"], df)
        return {"recommended_meal_ids": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Meal Recommendation API!"}
# Start the server by using uvicorn
# uvicorn main:app --reload