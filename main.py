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
            page_content=
            f"""Meal_Id: {row['meal_id']}\n
                Meal_Name: {row['meal_name']}\n
                Ingredient_Name: {row['ingredient_name']}\n
                Ingredient_Quantity: ({row['ingredient_qty']}\n
                Calo: ({row['calo']})""")
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
        You are a highly accurate and concise recommendation engine. 
        Based on the user's preferences below, suggest the most suitable meals:
        - Height: {height} cm
        - Weight: {weight} kg
        - Age: {age} years
        - Total daily calories: {calo}
        - Preferences: {preferences}
        - Lifestyle: {lifestyle}
        - Diet: {diet}
        - Allergies: {allergy}
        - Previous meals ID: {previous_meal_ids}
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
            "lifestyle", "diet", "allergy", "previous_meal_ids"
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
        previous_meal_ids=", ".join(map(str, preferences.get("previous_meal_ids", [])))
    )
def generate_question_one(preferences):
    """Generate a structured prompt using PromptTemplate."""
    template = """
        You are a highly accurate and concise recommendation engine.
        Based on the user's preferences below, suggest the most suitable meals:
        - Height: {height} cm
        - Weight: {weight} kg
        - Age: {age} years
        - Preferences: {preferences}
        - Lifestyle: {lifestyle}
        - Diet: {diet}
        - Allergies: {allergy}
        - Previous meal ids: {previous_meal_ids}
        - Target calories: {target_calo}
        Important:
        - Only return list meal IDs that exist in the provided dataset and is not in list {previous_meal_ids}
        - Do not create new IDs.
        - Ensure the meals meet the user's dietary requirements and preferences.
        - The calories of the meal need in range: 0 < {target_calo} - 100 <= meal calories <= {target_calo} + 100
        Return a list of unique meal IDs that meet the user's dietary and allergy requirements
        Do not include any explanation or additional text in the response.
        """
    prompt = PromptTemplate(
        input_variables=[
            "height", "weight", "age", "preferences",
            "lifestyle", "diet", "allergy", "previous_meals", "previous_meal_ids", "target_calo"
        ],
        template=template
    )
    return prompt.format(
        height=preferences["height"],
        weight=preferences["weight"],
        age=preferences["age"],
        preferences=preferences["preferences"],
        lifestyle=preferences["lifestyle"],
        diet=preferences["diet"],
        allergy=preferences["allergy"],
        previous_meal_ids=", ".join(map(str, preferences.get("previous_meal_ids", []))),
        target_calo=preferences["target_calo"],
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
    previous_meal_ids: List[int] = [],
    target_calo: Optional[int] = None,

def format_output(raw_output, df):
    """Extract a list of unique integer meal IDs from the model's raw output."""
    try:
        # Find array
        matches = re.findall(r"\[([^\]]+)\]", raw_output)
        if not matches:
            return []

        array_content = matches[0]
        meal_ids = [int(item.strip()) for item in array_content.split(",") if item.strip().isdigit()]
        # check id in data frame
        valid_meal_ids = [meal_id for meal_id in set(meal_ids)
                               if meal_id in df['meal_id'].values]
        if not valid_meal_ids:
            return []

        return valid_meal_ids
    except Exception as e:
        raise ValueError(f"Error parsing output: {str(e)}")

def get_combinations(arr, n):
    """Generate all combinations of length n from the list arr."""
    if n == 0:
        return [[]]
    if len(arr) < n:
        return []

    result = []
    for i in range(len(arr)):
        # For each element, get combinations from the rest of the list
        for rest in get_combinations(arr[i+1:], n-1):
            result.append([arr[i]] + rest)
    return result

def calc_meal_calo(meal_ids, total_calo, df, calo_diff=100):
    meal_calo = {meal_id: df[df['meal_id'] == meal_id]['calo'].values[0] for meal_id in meal_ids}
    def get_total_calo(ids):
        return sum(meal_calo[meal_id] for meal_id in ids)
    for i in range(3, len(meal_ids)+1):
        combinations = get_combinations(meal_ids, i)
        for comb in combinations:
            total = get_total_calo(comb)
            if total_calo - calo_diff <= total <= total_calo + calo_diff:
                return comb
    return [meal_ids[0], meal_ids[1], meal_ids[2]]

@app.post("/recommend")
async def recommend_meals(preferences: Preferences, file_path: str = "meal_ingredients.csv"):
    try:
        documents, df = load_data_from_csv(file_path)
        # Create RetrievalQA chain
        groq_api_key = api_key
        model_name = "llama3-8b-8192"
        qa_chain = create_retrieval_chain(documents, model_name, groq_api_key, preferences)
        # Generate question
        question = generate_question(preferences.model_dump())
        # Query the model
        raw_result = qa_chain.invoke(question)
        valid_meal_ids = format_output(raw_result["result"], df)
        return {
                "result": valid_meal_ids,
                "raw_result": raw_result["result"]
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/new")
async def recommend_meals_with_calo(
        preferences: Preferences,
        file_path: str = "meal_ingredients.csv"
    ):
    try:
        documents, df = load_data_from_csv(file_path)
        groq_api_key = api_key
        model_name = "llama3-8b-8192"
        qa_chain = create_retrieval_chain(documents, model_name, groq_api_key, preferences)
        preferences_dict = preferences.model_dump()
        question = generate_question_one(preferences_dict)
        raw_result = qa_chain.invoke(question)
        raw_meal_ids = format_output(raw_result["result"], df)
        target_calo = preferences_dict.get("target_calo", None)
        valid_meal_ids = []
        if preferences_dict.get("previous_meal_ids"):
            previous_meal_ids = set(preferences_dict["previous_meal_ids"])
            valid_meal_ids = [meal_id for meal_id in raw_meal_ids if meal_id not in previous_meal_ids]
        if preferences_dict["target_calo"]:
            valid_meal_ids = [
                meal_id for meal_id in raw_meal_ids
                if abs(int(df[df['meal_id'] == meal_id]['calo'].values[0]) - target_calo) <= 100
            ]
            min_meal_id = min(
                raw_meal_ids,
                key=lambda meal_id: abs(int(df[df['meal_id'] == meal_id]['calo'].values[0]) - target_calo)
            )
            if not valid_meal_ids:
                valid_meal_ids.append(min_meal_id)
        return {
            "result": valid_meal_ids,
            "raw_result": raw_result["result"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/day")
async def recommend_meals(preferences: Preferences, file_path: str = "meal_ingredients.csv"):
    try:
        documents, df = load_data_from_csv(file_path)
        groq_api_key = api_key
        model_name = "llama3-8b-8192"
        qa_chain = create_retrieval_chain(documents, model_name, groq_api_key, preferences)
        preferences_dict = preferences.model_dump()
        question = generate_question(preferences_dict)
        raw_result = qa_chain.invoke(question)
        valid_meal_ids = format_output(raw_result["result"], df)
        daily_meal_ids = calc_meal_calo(valid_meal_ids, preferences_dict["calo"],df, 150)
        return {
            "result": daily_meal_ids,
            "raw_result": raw_result["result"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/calo")
async def recommend_meals_with_total_calo(
        preferences: Preferences,
        file_path: str = "meal_ingredients.csv"
    ):
    try:
        documents, df = load_data_from_csv(file_path)
        groq_api_key = api_key
        model_name = "llama3-8b-8192"
        qa_chain = create_retrieval_chain(documents, model_name, groq_api_key, preferences)
        preferences_dict = preferences.model_dump()
        question = generate_question(preferences_dict)
        raw_result = qa_chain.invoke(question)
        raw_meal_ids = format_output(raw_result["result"], df)
        total_calo = preferences_dict["calo"]
        selected_meals = []
        total_calories = 0
        for meal_id in raw_meal_ids:
            calo = int(df[df['meal_id'] == meal_id]['calo'].values[0])
            if total_calories + calo <= total_calo - 100:
                selected_meals.append(meal_id)
                total_calories += calo
            if total_calories >= (total_calo - 100):
                break
        if total_calories < (total_calo - 100) and raw_meal_ids:
            remaining_calo = total_calo - total_calories
            closest_meal_id = min(
                (meal_id for meal_id in raw_meal_ids if meal_id not in selected_meals),
                key=lambda meal_id: abs(int(df[df['meal_id'] == meal_id]['calo'].values[0]) - remaining_calo)
            )
            selected_meals.append(closest_meal_id)
            total_calories += int(df[df['meal_id'] == closest_meal_id]['calo'].values[0])

        return {
            "result": selected_meals,
            "total_calo": total_calories,
            "raw_result": raw_meal_ids,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Welcome to the Meal Recommendation API!"}
# Start the server by using uvicorn
# uvicorn main:app --reload
