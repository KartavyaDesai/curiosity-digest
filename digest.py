import os
import sys
import json
import re
import hashlib
import logging
import smtplib
import time
import html as html_lib
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from email.message import EmailMessage
from logging.handlers import RotatingFileHandler
from typing import List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Console handler -> shows up in `Actions` run logs.
# Rotating file handler -> persists across runs if you cache/upload the file,
# and gives you full tracebacks instead of one-line summaries.

LOG_PATH = os.environ.get("DIGEST_LOG_PATH", "digest.log")

logger = logging.getLogger("digest")
logger.setLevel(logging.DEBUG)

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

_file = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
_file.setLevel(logging.DEBUG)
_file.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s [%(funcName)s] %(message)s")
)

logger.addHandler(_console)
logger.addHandler(_file)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def require_env(name: str) -> str:
    """Fail fast with a clear, traceable error instead of a bare KeyError."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Check your workflow's `env:` block / repo secrets."
        )
    return value


try:
    TYPESAFE_API_KEY = require_env("TYPESAFE_API_KEY")
    SMTP_HOST = require_env("SMTP_HOST")
    SMTP_USER = require_env("SMTP_USER")
    SMTP_PASSWORD = require_env("SMTP_PASSWORD")
except RuntimeError as exc:
    # Logging isn't fully useful yet (no run stats), but we still want this
    # traced clearly in both console and file before we exit.
    logger.critical("Startup configuration error: %s", exc)
    sys.exit(2)

TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-latest"

MAIL_TO = os.environ.get("MAIL_TO", "kartavyadesai555@gmail.com")
MAIL_FROM = os.environ.get("SMTP_USER", MAIL_TO)
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))

USER_PROFILE = os.environ.get(
    "USER_PROFILE",
    (
        "Primary interests: Software Engineering, Developer Tools, AI/ML/LLMs, "
        "Data Science, Research, Tech Startups, SaaS, emerging technology trends, "
        "technical articles and non-fiction. "
        "Avoid: clickbait, sponsored content, celebrity news, crypto hype, "
        "political outrage."
    ),
)

MAX_ARTICLES_PER_SOURCE = 5
RELEVANCE_THRESHOLD = 0.7
MAX_EMAIL_ARTICLES = 5
API_TIMEOUT_SECONDS = 20
API_MAX_RETRIES = 1
API_RETRY_BACKOFF_SECONDS = 2


@dataclass
class Source:
    name: str
    url: str
    category: str = "general"
    max_items: int = MAX_ARTICLES_PER_SOURCE


@dataclass
class Article:
    title: str
    url: str
    source: str
    description: str = ""
    category: str = "general"
    relevance_probability: float = 0.0
    quality_score: int = 0


@dataclass
class RunStats:
    """Collects everything worth knowing about a run for tracing/debugging."""
    sources_loaded: int = 0
    sources_failed: List[str] = field(default_factory=list)
    articles_fetched: int = 0
    articles_after_dedup: int = 0
    articles_eval_failed: int = 0
    articles_relevant: int = 0
    errors: List[str] = field(default_factory=list)

    def note_error(self, context: str, exc: Exception) -> None:
        msg = f"{context}: {type(exc).__name__}: {exc}"
        self.errors.append(msg)
        logger.exception(msg)  # logs full traceback to file + console

    def summary_lines(self) -> List[str]:
        return [
            f"Sources loaded: {self.sources_loaded} (failed: {len(self.sources_failed)})",
            f"Articles fetched: {self.articles_fetched}",
            f"Articles after dedup: {self.articles_after_dedup}",
            f"Articles that failed evaluation: {self.articles_eval_failed}",
            f"Relevant articles selected: {self.articles_relevant}",
            f"Total errors logged: {len(self.errors)}",
        ]


# ---------------------------------------------------------------------------
# Loading & fetching
# ---------------------------------------------------------------------------

def load_sources(stats: RunStats, path: str = "sources.json") -> List[Source]:
    try:
        with open(path, encoding="utf-8") as file:
            raw = json.load(file)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{path} not found next to the script") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} is not valid JSON: {exc}") from exc

    sources = []
    for i, item in enumerate(raw):
        try:
            sources.append(
                Source(
                    name=item["name"],
                    url=item["url"],
                    category=item.get("category", "general"),
                    max_items=int(item.get("max_items", MAX_ARTICLES_PER_SOURCE)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            stats.note_error(f"sources.json entry #{i} malformed: {item!r}", exc)

    stats.sources_loaded = len(sources)
    if not sources:
        raise RuntimeError("No valid sources loaded from sources.json")
    return sources


def clean_html(text: str) -> str:
    if not text:
        return ""
    try:
        text = BeautifulSoup(html_lib.unescape(text), "html.parser").get_text(" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        # Never let a malformed snippet of HTML kill the whole run.
        logger.warning("clean_html failed on a snippet; falling back to raw text", exc_info=True)
        return text.strip()


def fetch_articles(sources: List[Source], stats: RunStats) -> List[Article]:
    articles = []

    for source in sources:
        try:
            feed = feedparser.parse(source.url)

            # feedparser doesn't raise on network/parse errors, it sets `bozo`.
            if getattr(feed, "bozo", 0) and not feed.entries:
                raise RuntimeError(
                    f"feed parse error: {getattr(feed, 'bozo_exception', 'unknown')}"
                )

            count = 0
            for entry in feed.entries[: source.max_items]:
                title = clean_html(getattr(entry, "title", ""))
                url = getattr(entry, "link", "").strip()
                description = clean_html(
                    getattr(entry, "summary", "") or getattr(entry, "description", "")
                )

                if title and url:
                    articles.append(
                        Article(
                            title=title,
                            url=url,
                            source=source.name,
                            description=description,
                            category=source.category,
                        )
                    )
                    count += 1

            logger.info("Fetched %d articles from %s", count, source.name)

        except Exception as exc:
            stats.sources_failed.append(source.name)
            stats.note_error(f"Failed to fetch source '{source.name}' ({source.url})", exc)
            continue  # one bad source shouldn't stop the rest

    stats.articles_fetched = len(articles)
    return articles


def deduplicate(articles: List[Article], stats: RunStats) -> List[Article]:
    seen = set()
    unique = []

    for article in articles:
        try:
            normalized = re.sub(r"[^a-z0-9 ]", "", article.title.lower()).strip()
            digest = hashlib.sha256(normalized.encode()).hexdigest()

            if digest in seen:
                continue

            if any(
                SequenceMatcher(None, article.title.lower(), existing.title.lower()).ratio() >= 0.82
                for existing in unique
            ):
                continue

            seen.add(digest)
            unique.append(article)
        except Exception as exc:
            stats.note_error(f"Dedup failed for article '{article.title[:60]}'", exc)
            # Keep the article rather than silently dropping it on a dedup bug.
            unique.append(article)

    stats.articles_after_dedup = len(unique)
    return unique


# ---------------------------------------------------------------------------
# Evaluation (Typesafe / Jev)
# ---------------------------------------------------------------------------

def query_jev(article: Article) -> dict:
    payload = {
        "model": TYPESAFE_MODEL,
        "state": {
            "user_profile": USER_PROFILE,
            "article_title": article.title,
            "article_description": article.description[:1000],
        },
        "questions": {
            "is_relevant": {
                "type": "noul",
                "instructions": (
                    "Is this article relevant to the user's stated interests?"
                ),
                "criteria": {
                    "true": "The article meaningfully matches one or more of the user's interests.",
                    "false": "The article does not meaningfully match the user's interests."
                }
            },
            "quality_score": {
                "type": "score",
                "instructions": (
                    "How informative and useful is this article for a technically "
                    "knowledgeable reader?"
                ),
                "criteria": [
                    "Very poor",
                    "Poor",
                    "Below average",
                    "Average",
                    "Good",
                    "Very good",
                    "Excellent"
                ]
            }
            }
        },
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, API_MAX_RETRIES + 1):  # e.g. 3 total attempts
        try:
            response = requests.post(
                TYPESAFE_ENDPOINT,
                json=payload,
                headers={
                    "Authorization": f"Bearer {TYPESAFE_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=API_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt <= API_MAX_RETRIES:
                logger.warning(
                    "Jev API call failed for '%s' (attempt %d/%d): %s — retrying in %ds",
                    article.title[:60], attempt, API_MAX_RETRIES + 1, exc,
                    API_RETRY_BACKOFF_SECONDS,
                )
                time.sleep(API_RETRY_BACKOFF_SECONDS)
            else:
                break

    assert last_exc is not None
    raise last_exc


def evaluate_articles(articles: List[Article], stats: RunStats) -> List[Article]:
    relevant = []

    for article in articles:
        try:
            result = query_jev(article)
            answers = result.get("answers", {})

            probability = float(answers.get("is_relevant", {}).get("noul", 0.0))
            quality = int(answers.get("quality_score", {}).get("score", 0))

            article.relevance_probability = probability
            article.quality_score = quality

            if probability >= RELEVANCE_THRESHOLD:
                relevant.append(article)

        except requests.RequestException as exc:
            stats.articles_eval_failed += 1
            stats.note_error(f"Jev API error for '{article.title[:60]}' ({article.url})", exc)
        except (ValueError, TypeError, KeyError) as exc:
            stats.articles_eval_failed += 1
            stats.note_error(
                f"Malformed Jev response for '{article.title[:60]}' ({article.url})", exc
            )

    stats.articles_relevant = len(relevant)

    return sorted(
        relevant,
        key=lambda a: (a.relevance_probability * 0.7 + (a.quality_score / 10.0) * 0.3),
        reverse=True,
    )[:MAX_EMAIL_ARTICLES]


# ---------------------------------------------------------------------------
# Email building & sending
# ---------------------------------------------------------------------------

def build_debug_footer(stats: RunStats) -> str:
    lines = "".join(f"<li>{html_lib.escape(line)}</li>" for line in stats.summary_lines())
    error_block = ""
    if stats.errors:
        shown = stats.errors[:5]
        error_items = "".join(f"<li>{html_lib.escape(e)}</li>" for e in shown)
        more = f"<li>...and {len(stats.errors) - 5} more (see log file)</li>" if len(stats.errors) > 5 else ""
        error_block = f"""
        <p style="margin-top:12px;color:#b45309;font-weight:bold;">Errors during this run:</p>
        <ul style="font-size:12px;color:#b45309;">{error_items}{more}</ul>
        """
    return f"""
    <div style="margin-top:24px;padding:12px;background:#f6f6f6;border-radius:6px;">
      <p style="font-size:12px;color:#888;margin:0 0 6px;">Run summary</p>
      <ul style="font-size:12px;color:#888;margin:0;">{lines}</ul>
      {error_block}
    </div>
    """


def build_email(articles: List[Article], stats: RunStats) -> str:
    footer = build_debug_footer(stats)

    if not articles:
        return f"""
        <html>
          <body style="font-family:Arial,sans-serif;max-width:720px;margin:40px auto;padding:0 20px;color:#222;">
            <h2>Daily Research Digest</h2>
            <p>No relevant articles were found today.</p>
            {footer}
          </body>
        </html>
        """

    items = []
    for article in articles:
        summary = article.description[:350]
        if len(article.description) > 350:
            summary += "..."

        items.append(
            f"""
            <article style="margin-bottom:28px;">
              <h3 style="margin-bottom:6px;">
                <a href="{article.url}" style="color:#111;text-decoration:none;">
                  {html_lib.escape(article.title)}
                </a>
              </h3>
              <div style="color:#666;font-size:13px;margin-bottom:8px;">
                {html_lib.escape(article.source)}
                · Relevance {article.relevance_probability:.0%}
                · Quality {article.quality_score}/10
              </div>
              <p style="line-height:1.55;margin-top:0;">
                {html_lib.escape(summary)}
              </p>
            </article>
            """
        )

    return f"""
    <!doctype html>
    <html>
      <body style="font-family:Arial,sans-serif;max-width:720px;margin:40px auto;padding:0 20px;color:#222;">
        <h1 style="margin-bottom:4px;">Daily Research Digest</h1>
        <p style="color:#666;margin-top:0;">
          {len(articles)} relevant articles selected from today's sources.
        </p>
        {"".join(items)}
        <hr style="border:0;border-top:1px solid #ddd;margin-top:30px;">
        {footer}
        <p style="font-size:12px;color:#888;margin-top:8px;">
          Generated automatically with TypeSafe Jev.
        </p>
      </body>
    </html>
    """


def send_email(subject: str, html_content: str, plain_fallback: str = "") -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = MAIL_FROM
    message["To"] = MAIL_TO
    message.set_content(plain_fallback or "Your email client does not support HTML.")
    message.add_alternative(html_content, subtype="html")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(message)


def send_failure_alert(context: str, exc: Exception) -> None:
    """
    Best-effort notification when the run fails hard, so a broken pipeline
    doesn't just go silent. Failures here are logged but not re-raised —
    the original error is what should determine the exit code.
    """
    try:
        body = (
            f"<html><body style='font-family:Arial,sans-serif;'>"
            f"<h2>Daily Research Digest — run failed</h2>"
            f"<p><b>Where:</b> {html_lib.escape(context)}</p>"
            f"<p><b>Error:</b> {html_lib.escape(f'{type(exc).__name__}: {exc}')}</p>"
            f"<p>Check the workflow run logs / {html_lib.escape(LOG_PATH)} for the full traceback.</p>"
            f"</body></html>"
        )
        send_email("Daily Research Digest — FAILED", body, plain_fallback=f"Run failed at {context}: {exc}")
        logger.info("Failure alert email sent.")
    except Exception:
        logger.exception("Also failed to send the failure-alert email. Check SMTP config/log file directly.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    stats = RunStats()
    logger.info("=== Digest run starting ===")

    try:
        sources = load_sources(stats)
        articles = fetch_articles(sources, stats)
        articles = deduplicate(articles, stats)
        relevant_articles = evaluate_articles(articles, stats)
        html_content = build_email(relevant_articles, stats)
        send_email("Daily Research Digest", html_content)

        logger.info("=== Digest run finished ===")
        for line in stats.summary_lines():
            logger.info(line)

        # Non-fatal errors happened but the run still completed and sent mail.
        if stats.errors:
            logger.warning("Run completed with %d non-fatal errors logged above.", len(stats.errors))
        return 0

    except Exception as exc:
        # Anything that escapes here is fatal: config, sources.json, or SMTP itself broke.
        logger.critical("Fatal error, run aborted: %s", exc, exc_info=True)
        send_failure_alert("main()", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
