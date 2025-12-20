import os
import uvicorn
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# 1. SETUP: Load Environment Variables
load_dotenv()

# 2. PATHS: Locate the templates folder relative to this file
# This assumes app.py is in /src/api/
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
# We check project root first, then local api folder
TEMPLATE_DIR = PROJECT_ROOT / "templates" 
if not TEMPLATE_DIR.exists():
    TEMPLATE_DIR = CURRENT_DIR / "templates"

print(f"--- System Booting ---")
print(f"Templates Path: {TEMPLATE_DIR}")
print(f"Gemini Key Loaded: {'Yes' if os.getenv('GEMINI_API_KEY') else 'No'}")

# 3. IMPORT CUSTOM LOGIC
from src.inference.predictor import predict_query
from src.notifications.gemini_email import generate_email
from src.notifications.email_sender import send_email
from src.config.team_emails import TEAM_EMAILS

# 4. INITIALIZE APP
app = FastAPI(title="Moderne AI Resolution Desk")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "warning": None, "status": None}
    )

@app.post("/submit", response_class=HTMLResponse)
async def submit_query(request: Request, user_query: str = Form(...)):
    # AI Step 1: Internal Classification
    intent, team = predict_query(user_query)

    # Handle Unclear Intent
    if intent == "Unclear":
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "warning": True,
                "error_message": "The inquiry details are insufficient for automated routing. Please provide more context.",
                "query": user_query
            }
        )

    # AI Step 2: Gemini Elaboration & Email Dispatch
    try:
        # Calls the function we fixed with response.text
        email_body = generate_email(user_query, intent, team)
        
        receiver_email = TEAM_EMAILS.get(team)
        if receiver_email:
            send_email(
                to_email=receiver_email,
                subject=f"New Ticket Assignment – {intent}",
                body=email_body
            )
            status_msg = "Successfully routed to department."
        else:
            status_msg = "Routed, but department email not configured."

    except Exception as e:
        print(f"System Error: {e}")
        status_msg = "Routing error. Logged for manual review."

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "query": user_query,
            "intent": intent,
            "team": team,
            "status": status_msg,
            "warning": False
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)