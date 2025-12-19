import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.0-flash"

# Create model once (important)
model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={
        "temperature": 0.2,
        "max_output_tokens": 120
    }
)


def elaborate_query_gemini(user_query: str, intent: str) -> str:
    """Uses Gemini to rewrite the query into a professional summary."""

    prompt = f"""
You are an expert internal support coordinator for a luxury retail group.

TASK: Convert the following customer query into a highly professional internal support summary.

INPUT DETAILS:
- User Intent: {intent}
- User Query: "{user_query}"

GUIDELINES:
1. Rewrite it in exactly 2 sentences of polished, business-grade English.
2. Maintain the original facts exactly.
3. Use a tone suitable for a premium brand.
4. Focus on the required action for the internal team.

RESPONSE FORMAT:
Provide ONLY the rewritten text.
"""

    try:
        response = model.generate_content(prompt)

        # ✅ SAFE EXTRACTION (this is the fix)
        if response and response.candidates:
            parts = response.candidates[0].content.parts
            if parts:
                return parts[0].text.strip()

        return user_query  # fallback if Gemini returns empty

    except Exception as e:
        return f"[Manual Review Required] {user_query}"


def generate_email(user_query: str, intent: str, team: str) -> str:
    """Combines Gemini summary into an internal email"""

    professional_summary = elaborate_query_gemini(user_query, intent)

    error_note = ""
    if professional_summary.startswith("[Manual Review Required]"):
        error_note = "\n(Note: Automated elaboration service is temporarily unavailable.)\n"

    email = f"""
NEW SERVICE INQUIRY | {intent.upper()}
--------------------------------------------------
DEAR {team.upper()},

A formal inquiry has been triaged and routed to your department.

SUMMARY OF ISSUE:
{professional_summary}
{error_note}

ORIGINAL CUSTOMER STATEMENT:
"{user_query}"

TECHNICAL DETAILS:
- Classification: {intent}

Please acknowledge this case and initiate the resolution process within the next business hour.

REGARDS,
MODERNE RESOLUTION SYSTEM
--------------------------------------------------
"""
    return email.strip()
