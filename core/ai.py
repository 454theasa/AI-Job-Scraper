"""Central AI configuration: resolves the Gemini API key + model per profile.

Priority for the key: the active profile's key stored in the DB, then the
GEMINI_API_KEY environment variable. The model comes from the profile, with
DEFAULT_MODEL as fallback.
"""
import os
from typing import Optional, Tuple

from google import genai
from sqlmodel import Session

from .models import User, UserContext

DEFAULT_MODEL = "gemini-2.5-flash"

class MissingApiKeyError(ValueError):
    pass

def resolve_ai_config_for_user(session: Session, user_id: Optional[int]) -> Tuple[Optional[str], str]:
    """Returns (api_key, model_name) for a User. api_key may be None."""
    api_key = None
    model = DEFAULT_MODEL

    if user_id is not None:
        user = session.get(User, user_id)
        if user is not None:
            if user.gemini_api_key:
                api_key = user.gemini_api_key
            if user.gemini_model:
                model = user.gemini_model

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    return api_key, model

def resolve_ai_config(session: Session, user_context_id: Optional[int]) -> Tuple[Optional[str], str]:
    """Returns (api_key, model_name). api_key may be None if nothing is configured."""
    user_id = None
    if user_context_id is not None:
        ctx = session.get(UserContext, user_context_id)
        if ctx is not None:
            user_id = ctx.user_id
    return resolve_ai_config_for_user(session, user_id)

def get_genai_client(session: Session, user_context_id: Optional[int]):
    """Returns (client, model_name). Raises MissingApiKeyError with a
    user-friendly message when no key is configured anywhere."""
    api_key, model = resolve_ai_config(session, user_context_id)
    if not api_key:
        raise MissingApiKeyError(
            "No Gemini API key configured for this profile. "
            "Open the '🔑 AI Settings' panel in the sidebar and paste your key, "
            "or set the GEMINI_API_KEY environment variable."
        )
    return genai.Client(api_key=api_key), model
