"""Weekly paper alert sender.

Reads alert subscriptions from Supabase, finds papers published in the last
7 days that match each saved topic (vector similarity search), and emails a
personalised digest via SendGrid.

Required environment variables:
    SUPABASE_URL        – Supabase project URL
    SUPABASE_KEY        – Supabase service-role key
    SENDGRID_API_KEY    – SendGrid API key
    ALERT_FROM_EMAIL    – Verified sender address (default: alerts@arxivrag.app)

Run manually:
    python3 send_alerts.py

Triggered automatically every Monday at 08:00 UTC via GitHub Actions
(.github/workflows/paper_alerts.yml).
"""

import datetime
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client
import sendgrid
from sendgrid.helpers.mail import Mail

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
SENDGRID_KEY = os.environ["SENDGRID_API_KEY"]
FROM_EMAIL   = os.environ.get("ALERT_FROM_EMAIL", "alerts@arxivrag.app")
LOOKBACK_DAYS = 7
MATCH_THRESHOLD = 0.40
MATCH_COUNT = 20

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
model    = SentenceTransformer("all-MiniLM-L6-v2")


# ── helpers ───────────────────────────────────────────────────────────────────

def fetch_subscriptions() -> list[dict]:
    resp = supabase.table("paper_alerts").select("*").execute()
    return resp.data or []


def fetch_new_papers_for_topic(topic: str, cutoff: str) -> list[dict]:
    """Return papers published on or after `cutoff` (ISO date string) that
    match `topic` via vector similarity."""
    vec = model.encode(topic).tolist()
    try:
        resp = supabase.rpc("match_documents", {
            "query_embedding": vec,
            "match_threshold": MATCH_THRESHOLD,
            "match_count":     MATCH_COUNT,
        }).execute()
    except Exception as e:
        print(f"  Retrieval error for topic '{topic}': {e}")
        return []

    seen:   set[str]  = set()
    papers: list[dict] = []
    for doc in (resp.data or []):
        meta      = doc.get("metadata", {})
        title     = meta.get("title", "")
        published = meta.get("published", "")
        if title and title not in seen and published >= cutoff:
            seen.add(title)
            papers.append(meta)
    return papers


def build_email_html(topics_papers: dict[str, list[dict]]) -> str:
    lines = [
        "<html><body>",
        "<h2 style='font-family:sans-serif;'>Your ArXiv Weekly Research Digest</h2>",
        "<p style='font-family:sans-serif;color:#555;'>Papers from the last 7 days "
        "matching your saved topics.</p>",
    ]
    for topic, papers in topics_papers.items():
        lines.append(
            f"<h3 style='font-family:sans-serif;border-bottom:1px solid #eee;"
            f"padding-bottom:4px;'>{topic}</h3>"
        )
        if papers:
            for p in papers:
                title = p.get("title", "Unknown")
                url   = p.get("url", "#")
                pub   = p.get("published", "")[:10]
                cat   = p.get("category", "")
                lines.append(
                    f'<p style="font-family:sans-serif;margin:6px 0;">'
                    f'• <a href="{url}">{title}</a> '
                    f'<span style="color:#888;">({cat} · {pub})</span></p>'
                )
        else:
            lines.append(
                '<p style="font-family:sans-serif;color:#888;">'
                "<em>No new papers this week.</em></p>"
            )
    lines.append("</body></html>")
    return "\n".join(lines)


def send_email(to_email: str, subject: str, html_body: str) -> None:
    sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_KEY)
    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=to_email,
        subject=subject,
        html_content=html_body,
    )
    sg.send(message)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cutoff = (
        datetime.datetime.utcnow() - datetime.timedelta(days=LOOKBACK_DAYS)
    ).date().isoformat()

    subs = fetch_subscriptions()
    print(f"Found {len(subs)} subscription(s). Cutoff: {cutoff}")

    for sub in subs:
        email  = sub.get("email", "").strip()
        topics = sub.get("topics") or []
        if not email or not topics:
            continue

        print(f"Processing {email} ({len(topics)} topic(s))…")
        topics_papers: dict[str, list[dict]] = {}
        for topic in topics:
            papers = fetch_new_papers_for_topic(topic, cutoff)
            topics_papers[topic] = papers
            print(f"  '{topic}' → {len(papers)} new paper(s)")

        html_body = build_email_html(topics_papers)
        try:
            send_email(email, "Your ArXiv Weekly Research Digest", html_body)
            print(f"  Sent to {email}")
        except Exception as e:
            print(f"  Failed to send to {email}: {e}")


if __name__ == "__main__":
    main()
