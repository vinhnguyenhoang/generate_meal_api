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
    
    retriever = vectorstore.as_retriever(filter_function=custom_filter)
    retriever.search_kwargs["k"] = 50 # Increase the number of returned documents
    return retriever
def create_retrieval_chain(documents, model_name, groq_api_key, preferences):
    """Create a RetrievalQA chain using the given documents and model."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = retriever = create_custom_retriever(vectorstore, preferences)


    model = ChatGroq(model=model_name, api_key=groq_api_key, temperature=0.3)
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
Important:
- Only return meal IDs that exist in the provided dataset. The array at least 9 id.
- Do not create new IDs.
- Ensure the meals meet the user's dietary requirements and preferences.
Return a list of unique meal IDs that meet the user's dietary and allergy requirements
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

def cal_calo(calo, meal_ids, df):
    """
    Group meal IDs into groups of 3 with total calories within the allowed range.
    """
    groups = []
    n = len(meal_ids)
    lower_bound = calo - 200
    upper_bound = calo + 200

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Get meal IDs
                meal_id_group = [meal_ids[i], meal_ids[j], meal_ids[k]]
                
                # Calculate total calories
                total_calo = sum(
                    df.loc[df["meal_id"] == meal_id, "calo"].values[0]
                    for meal_id in meal_id_group
                )
                
                # Check if within range
                if lower_bound <= total_calo <= upper_bound:
                    groups.append(meal_id_group)
    
    return groups

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
        valid_meal_ids = format_output(raw_result["result"], df)
        recommended_groups = cal_calo(preferences.calo, valid_meal_ids, df)
        return {"result": recommended_groups,
                "raw_result": raw_result
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Meal Recommendation API!"}
# Start the server by using uvicorn
# uvicorn main:app --reload