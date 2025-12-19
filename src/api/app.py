from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.inference.predictor import predict_query
from src.notifications.gemini_email import generate_email
from src.notifications.email_sender import send_email
from src.config.team_emails import TEAM_EMAILS

app = FastAPI(title="AI Ticket Routing System")

BASE_DIR = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "warning": None} # Initialize warning as None
    )

@app.post("/submit", response_class=HTMLResponse)
async def submit_query(request: Request, user_query: str = Form(...)):
    # 1. Triage the query
    intent, team = predict_query(user_query)

    # 🚨 Check for unclear intent
    if intent == "Unclear":
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "warning": True # Triggers the CSS warning block we added
            }
        )

    # ✅ If clear, proceed to Gemini Elaboration
    try:
        # Calling the Gemini-linked function
        email_body = generate_email(user_query, intent, team)
        
        receiver_email = TEAM_EMAILS.get(team)
        if receiver_email:
            send_email(
                to_email=receiver_email,
                subject=f"New Ticket – {intent}",
                body=email_body
            )
        status_msg = "Dispatch successful. Team notified."
    except Exception as e:
        print(f"System Error: {e}")
        status_msg = "Ticket logged, but notification failed. Manual review scheduled."

    # Return success state
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "query": user_query,
            "intent": intent,
            "team": team,
            "status": status_msg,
            "warning": False # Explicitly hide warning on success
        }
    )