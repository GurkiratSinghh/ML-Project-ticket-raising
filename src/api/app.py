from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.inference.predictor import predict_query
from src.notifications.hf_email import generate_email
from src.notifications.email_sender import send_email
from src.config.team_emails import TEAM_EMAILS


app = FastAPI(title="AI Ticket Routing System")

# Resolve base directory
BASE_DIR = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/submit", response_class=HTMLResponse)
def submit_query(request: Request, user_query: str = Form(...)):
    intent, team = predict_query(user_query)

    email_body = generate_email(user_query, intent, team)

    receiver_email = TEAM_EMAILS.get(team)
    if receiver_email:
        send_email(
            to_email=receiver_email,
            subject=f"New Ticket – {intent}",
            body=email_body
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "query": user_query,
            "intent": intent,
            "team": team,
            "status": "Email sent successfully"
        }
    )
