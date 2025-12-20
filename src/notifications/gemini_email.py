import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini with a safety check
if not GEMINI_API_KEY:
    print("CRITICAL: GEMINI_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.0-flash"

model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={
        "temperature": 0.2,
        "max_output_tokens": 150,
    }
)

def elaborate_query_gemini(user_query: str, intent: str) -> str:
    """Uses Gemini to rewrite the query into a professional summary with robust error handling."""
    
    prompt = f"""
    You are an expert internal support coordinator for a luxury retail group.
    TASK: Convert the following customer query into a highly professional internal support summary.
    
    INPUT:
    - Intent: {intent}
    - Query: "{user_query}"
    
    GUIDELINES:
    1. Rewrite it in exactly 2 sentences of polished, business-grade English.
    2. Maintain facts exactly. Use a premium, formal tone.
    3. Focus on the required action for the team.
    
    RESPONSE FORMAT:
    Provide ONLY the rewritten text. No introductions or conversational filler.
    """

    try:
        response = model.generate_content(prompt)
        
        # The .text property is the safest way to get content. 
        # If the response is blocked or empty, this will raise an AttributeError which we catch.
        if response and response.text:
            return response.text.strip()
        
        return user_query 

    except Exception as e:
        # Log the specific error for debugging if needed: print(f"Gemini Error: {e}")
        return f"[Manual Review Required] {user_query}"

def generate_email(user_query: str, intent: str, team: str) -> str:
    """Combines summary into a clean, formatted internal email template."""
    professional_summary = elaborate_query_gemini(user_query, intent)
    
    error_note = ""
    if professional_summary.startswith("[Manual Review Required]"):
        error_note = "\n⚠️ ALERT: Automated summary failed. Please review original text.\n"

    return f"""
NEW SERVICE INQUIRY | {intent.upper()}
--------------------------------------------------
DEAR {team.upper()} TEAM,

A formal inquiry has been triaged and routed to your department.

SUMMARY OF ISSUE:
{professional_summary}
{error_note}

ORIGINAL CUSTOMER STATEMENT:
"{user_query}"

TECHNICAL DETAILS:
- Classification: {intent}

Please acknowledge this case and initiate resolution within the service level agreement.

REGARDS,
MODERNE RESOLUTION SYSTEM
--------------------------------------------------
""".strip()