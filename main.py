import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.inference.predictor import predict_query
from src.notifications.hf_email import generate_email
from src.notifications.email_sender import send_email
from src.config.team_emails import TEAM_EMAILS


def print_header():
    print("\n" + "*" * 44)
    print("*      AI QUERY TICKET RAISING SYSTEM       *")
    print("*" * 44 + "\n")


def print_result(intent, team, email_status):
    print("\n" + "-" * 44)
    print(f" Predicted Intent       : {intent}")
    print(f" Assigned Team          : {team}")
    print(f" Email Status           : {email_status}")
    print("-" * 44 + "\n")


def main():
    print_header()

    user_query = input("👉 Enter customer query:\n> ").strip()

    if not user_query:
        print("\n❗ Query cannot be empty. Exiting.\n")
        return

    print("\n⏳ Processing query...\n")

    try:
        intent, team = predict_query(user_query)

        email_body = generate_email(user_query, intent, team)

        receiver_email = TEAM_EMAILS.get(team)

        if receiver_email:
            send_email(
                to_email=receiver_email,
                subject=f"New Ticket Assigned – {intent}",
                body=email_body
            )
            email_status = "Sent Successfully"
        else:
            email_status = "Team email not configured"

        print_result(intent, team, email_status)

    except Exception as e:
        print("\n❌ Something went wrong while processing the request.")
        print("Details:", e)


if __name__ == "__main__":
    main()
