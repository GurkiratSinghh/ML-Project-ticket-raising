from transformers import pipeline

rewriter = pipeline(
    "text2text-generation",
    model="google/flan-t5-large"
)


def generate_email(user_query: str, intent: str, team: str) -> str:
    prompt = f"""
Rewrite the following customer issue into TWO professional sentence
suitable for an internal support email.

Customer Issue:
{user_query}

Professional Version:
"""

    try:
        result = rewriter(
            prompt,
            max_length=80,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0
        )
        professional_issue = result[0]["generated_text"].strip()
    except Exception:
        professional_issue = user_query

    email = f"""
Subject: New Support Ticket – {intent}

Dear {team.title()},

A customer has raised the following request:

{professional_issue}

Identified Intent: {intent}

Please review the request and take the necessary action at the earliest.

Regards,
Customer Support System
"""

    return email.strip()
