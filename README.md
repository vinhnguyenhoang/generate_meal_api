# Meal Recommendation API

This is a FastAPI-based application that recommends meals based on user preferences such as height, weight, age, allergies, lifestyle, and previous meals. The application uses a machine learning model to retrieve the best meal suggestions from a dataset of meal ingredients.

## Prerequisites

Before you can run the app, you need to have the following installed:

- Python 3.8+ (preferably Python 3.10 or higher) (Ex: Python 3.10.7)
- `pip` (Python package installer)

## Setup Instructions

### 1. Clone the Repository

Clone this repository to your local machine:

```
    git clone <repository_url>
    cd <repository_directory>
```
### 2. Create a Virtual Environment
Create a virtual environment to manage dependencies:

```
python -m venv venv
```

Activate the virtual environment:

On Windows:
```
.\venv\Scripts\activate
```
On macOS/Linux:

```
source venv/bin/activate
```
### 3. Install Dependencies
Once the virtual environment is activated, install the required dependencies:


```
pip install -r requirements.txt
```
### 4. Prepare the Data
Make sure the meal_ingredients.csv file is present in the project directory. This file should contain meal data, including meal_id, meal_name, ingredient_name, and ingredient_qty.

### 5. Run the Application
Start the FastAPI server with Uvicorn:


```
uvicorn main:app --reload
```

By default, the app will be accessible at http://127.0.0.1:8000.

### 6. Test the API Using Postman
You can use Postman to test the API. Below are the details for testing the /recommend endpoint:

Endpoint: POST /recommend
This endpoint recommends meals based on the user's preferences. It expects a JSON payload in the body of the request.

Request body format:

```
{
    "height": 170,
    "weight": 70,
    "age": 25,
    "preferences": "Vegetarian",
    "lifestyle": "Active",
    "diet": "Low-carb",
    "allergy": "None",
    "previous_meals": ["Salad", "Grilled chicken"]
}
```
Example Request in Postman:
URL: http://127.0.0.1:8000/recommend
Method: POST
Body: Select the raw option and set the body type to JSON, then paste the sample JSON above.
Expected Response:
```
{
    "recommended_meal_ids": [123, 456, 789]
}
```
This will return an array of meal_ids that match the user's preferences.

### 7. Accessing the Documentation
FastAPI provides automatic API documentation. You can access the interactive API docs at the following URLs:

Swagger UI: http://127.0.0.1:8000/docs
ReDoc: http://127.0.0.1:8000/redoc
These pages will help you explore and test the available API endpoints.

Troubleshooting
Error: ModuleNotFoundError

Ensure that all dependencies are installed by running `pip install -r requirements.txt`.
Error: File not found

Make sure the meal_ingredients.csv file exists in the same directory as the app.

### Summary:
This `README.md` file provides step-by-step instructions for setting up the FastAPI app, running it local