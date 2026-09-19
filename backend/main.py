import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the Gemini client using the API key from the environment
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

app = FastAPI()

# Enable CORS so your frontend can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    resume_text: str
    job_description: str

@app.post("/api/optimize")
async def optimize_resume(data: OptimizeRequest):
    try:
        # Check if API key is loaded properly
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing from environment variables.")

        prompt = f"""
        You are an expert technical recruiter and resume writer. 
        Analyze the following Resume against the Target Job Description.
        
        Target Job Description:
        {data.job_description}

        Original Resume:
        {data.resume_text}

        Provide:
        1. An optimized, ATS-friendly version of the resume highlighting matching keywords.
        2. A brief bulleted changelog explaining what was changed and why.
        """

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        
        return {"result": response.text}
        
    except Exception as e:
        print(f"ERROR OCCURRED: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))