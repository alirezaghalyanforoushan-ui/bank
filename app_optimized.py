# -*- coding: utf-8 -*-
"""Streamlit UI for the Adaptive Narrative Learning Game.

Run with:
    streamlit run app_optimized.py
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any

import streamlit as st

from adaptive_narrative_learning_game_rag_powered import (
    BLOOM_LEVELS,
    GEMINI_TEXT_MODEL,
    GENERATION_PROVIDER_GEMINI,
    GENERATION_PROVIDER_LMSTUDIO,
    LMSTUDIO_BASE_URL,
    GameEngineError,
    GeminiRequestError,
    LMStudioRequestError,
    ResourceExtractionError,
    ResourceValidationError,
    discover_text_rich_resources,
    expand_retrieval_query,
    extract_subtopics,
    extract_uploaded_pdf,
    friendly_generation_error,
    generate_beat,
    generate_lmstudio_json,
    level_subtopics,
    list_lmstudio_models,
    next_bloom_index,
    normalize_lmstudio_base_url,
    validate_public_url,
)


st.set_page_config(
    page_title="Adaptive Narrative Learning Game",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SUPPORTED_MODELS = list(
    dict.fromkeys(
        [
            GEMINI_TEXT_MODEL,
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash",
        ]
    )
)


# ---------------------------------------------------------------------------
# Session and presentation helpers
# ---------------------------------------------------------------------------


def _secret_api_key() -> str:
    try:
        value = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        value = ""
    return str(value or os.getenv("GOOGLE_API_KEY", "")).strip()


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "step": "introduction",
        "language": "english",
        "provider": GENERATION_PROVIDER_GEMINI,
        "api_key": _secret_api_key(),
        "gemini_model": GEMINI_TEXT_MODEL,
        "lmstudio_base_url": LMSTUDIO_BASE_URL,
        "lmstudio_api_token": "",
        "lmstudio_models": [],
        "lmstudio_model": "",
        "setup": None,
        "candidate_subtopics": [],
        "selected_subtopics": [],
        "game": None,
        "current_question": None,
        "answer_result": None,
        "question_started_at": None,
        "font_px": 18,
        "time_limit": 100,
        "resource_search_domain": "",
        "resource_search_query": "",
        "resource_suggestions": [],
        "resource_suggestion_query": "",
        "uploaded_pdf_documents": [],
        "pdf_uploader_nonce": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def is_fa() -> bool:
    return st.session_state.get("language") == "persian"


def tr(english: str, persian: str) -> str:
    return persian if is_fa() else english


def goto(step: str) -> None:
    st.session_state.step = step
    st.rerun()


def reset_question_state() -> None:
    st.session_state.current_question = None
    st.session_state.answer_result = None
    st.session_state.question_started_at = None


def reset_game(*, keep_credentials: bool = True) -> None:
    preserved = {
        "language": st.session_state.get("language", "english"),
        "provider": st.session_state.get("provider", GENERATION_PROVIDER_GEMINI),
        "font_px": st.session_state.get("font_px", 18),
        "time_limit": st.session_state.get("time_limit", 100),
    }
    if keep_credentials:
        preserved.update(
            {
                "api_key": st.session_state.get("api_key", ""),
                "gemini_model": st.session_state.get("gemini_model", GEMINI_TEXT_MODEL),
                "lmstudio_base_url": st.session_state.get("lmstudio_base_url", LMSTUDIO_BASE_URL),
                "lmstudio_api_token": st.session_state.get("lmstudio_api_token", ""),
                "lmstudio_models": st.session_state.get("lmstudio_models", []),
                "lmstudio_model": st.session_state.get("lmstudio_model", ""),
            }
        )
    for session_key in list(st.session_state.keys()):
        del st.session_state[session_key]
    init_session_state()
    for key, value in preserved.items():
        st.session_state[key] = value


def active_model() -> str:
    if st.session_state.get("provider") == GENERATION_PROVIDER_LMSTUDIO:
        return str(st.session_state.get("lmstudio_model", "")).strip()
    return str(st.session_state.get("gemini_model", GEMINI_TEXT_MODEL)).strip()


def provider_label(provider: str | None = None) -> str:
    value = provider or st.session_state.get("provider", GENERATION_PROVIDER_GEMINI)
    if value == GENERATION_PROVIDER_LMSTUDIO:
        return tr("LM Studio (local)", "LM Studio (محلی)")
    return "Gemini API"

def apply_css() -> None:
    base = max(13, min(int(st.session_state.get("font_px", 18)), 28))
    direction = "rtl" if is_fa() else "ltr"
    alignment = "right" if is_fa() else "left"
    st.markdown(
        f"""
<style>
/*
Keep the browser shell LTR so the page scrollbar remains on the normal outer
edge. The Streamlit app container may still use RTL to place the sidebar on the
right in Persian mode. The sidebar shell itself is forced back to LTR so its
own scrollbar stays at the outside edge rather than between the sidebar and
main page.
*/
html, body {{ direction: ltr !important; }}
[data-testid="stAppViewContainer"] {{ direction: {direction} !important; }}

/* Main-page reading direction. */
[data-testid="stMainBlockContainer"],
[data-testid="stMainBlockContainer"] > div,
.main .block-container {{ direction: {direction}; }}

[data-testid="stMarkdownContainer"],
[data-testid="stAlert"],
[data-testid="stWidgetLabel"],
label {{ text-align: {alignment}; }}

/* Sidebar geometry and scrolling must remain LTR. */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {{
    direction: ltr !important;
}}
[data-testid="stSidebarContent"] {{
    overflow-x: hidden !important;
    overflow-y: auto !important;
    scrollbar-width: thin;
}}
[data-testid="stSidebarContent"]::-webkit-scrollbar {{ width: 8px; }}

/* Restore Persian direction only for the sidebar's visible text and fields. */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stAlert"],
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {{
    direction: {direction} !important;
    text-align: {alignment} !important;
}}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {{
    direction: {direction} !important;
    text-align: {alignment} !important;
}}

/* Do not leave an overflowing inner panel visible when the sidebar collapses. */
[data-testid="stSidebar"][aria-expanded="false"],
[data-testid="stSidebar"][aria-expanded="false"] > div,
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] {{
    overflow: hidden !important;
}}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p {{
    font-size: {base}px !important;
    line-height: 1.65;
}}
.stButton button, .stDownloadButton button {{ font-size: {base}px !important; }}
.game-card {{
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin: .5rem 0 1rem;
}}
.source-link {{ overflow-wrap: anywhere; }}
</style>
""",
        unsafe_allow_html=True,
    )


def sidebar_settings() -> None:
    with st.sidebar:
        st.header(tr("Display", "نمایش"))
        st.session_state.font_px = st.slider(
            tr("Font size", "اندازه فونت"),
            13,
            28,
            int(st.session_state.font_px),
            key="font_size_setting",
        )
        st.session_state.time_limit = st.slider(
            tr("Question time limit (seconds)", "مهلت هر سؤال (ثانیه)"),
            30,
            300,
            int(st.session_state.time_limit),
            step=10,
            key="time_limit_setting",
            help=tr(
                "The limit is checked when the answer is submitted; the page does not rerun every second.",
                "مهلت هنگام ثبت پاسخ بررسی می‌شود؛ صفحه هر ثانیه بازاجرا نمی‌شود.",
            ),
        )
        st.caption(
            tr(
                "Cloud API keys and local API tokens remain only in this browser session and are excluded from exported progress.",
                "کلیدهای ابری و Token محلی فقط در نشست مرورگر نگه‌داری می‌شوند و در خروجی پیشرفت ذخیره نمی‌شوند.",
            )
        )


def show_engine_error(exc: Exception) -> None:
    if isinstance(exc, (GeminiRequestError, LMStudioRequestError)):
        st.error(friendly_generation_error(exc, st.session_state.language))
    elif isinstance(exc, ResourceValidationError):
        st.error(tr(f"Invalid resource: {exc}", f"منبع نامعتبر است: {exc}"))
    elif isinstance(exc, ResourceExtractionError):
        st.error(
            tr(
                f"Could not extract usable text from the resources: {exc}",
                f"متن آموزشی قابل‌استفاده از منابع استخراج نشد: {exc}",
            )
        )
    elif isinstance(exc, GameEngineError):
        st.error(str(exc))
    else:
        st.error(tr(f"Unexpected error: {exc}", f"خطای پیش‌بینی‌نشده: {exc}"))


# ---------------------------------------------------------------------------
# Source analysis
# ---------------------------------------------------------------------------


def configured_extract_subtopics(
    domain: str,
    urls: tuple[str, ...],
    language: str,
    uploaded_documents: list[dict[str, Any]],
) -> list[str]:
    """Run one extraction request without caching credentials or local tokens."""
    provider = st.session_state.provider
    return extract_subtopics(
        domain,
        urls,
        language,
        uploaded_documents=uploaded_documents,
        provider=provider,
        api_key=(
            st.session_state.api_key
            if provider == GENERATION_PROVIDER_GEMINI
            else None
        ),
        model=active_model(),
        lmstudio_base_url=(
            st.session_state.lmstudio_base_url
            if provider == GENERATION_PROVIDER_LMSTUDIO
            else None
        ),
        lmstudio_api_token=(
            st.session_state.lmstudio_api_token
            if provider == GENERATION_PROVIDER_LMSTUDIO
            else None
        ),
    )


@st.cache_data(show_spinner=False, max_entries=48)
def cached_extract_uploaded_pdf(filename: str, content: bytes) -> dict[str, Any]:
    return extract_uploaded_pdf(filename, content)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def introduction_page() -> None:
    st.title("🎮 Adaptive Narrative Learning Game")
    st.markdown(
        tr(
            """
Turn trusted educational webpages or PDFs into a source-grounded, adaptive
learning game. Generate with **Gemini API** or run fully locally through
**LM Studio** at `http://127.0.0.1:1234`.

**Highlights**

- Local BM25 retrieval uses no embedding API.
- Gemini is called only for curriculum and question generation.
- LM Studio mode needs no Google key and sends prompts only to your local server.
- JSON Schema output is requested for reliable questions, with a compatibility fallback.
- Public source URLs are protected against local-network access.
- Per-topic progress, response time, scoring, and Bloom history are tracked accurately.
""",
            """
صفحه‌های آموزشی یا PDF معتبر را به یک بازی یادگیری تطبیقی و مبتنی بر منبع
تبدیل کنید. تولید محتوا می‌تواند با **Gemini API** یا کاملاً محلی از طریق
**LM Studio** روی `http://127.0.0.1:1234` انجام شود.

**ویژگی‌ها**

- بازیابی محلی BM25 هیچ سهمیه embedding مصرف نمی‌کند.
- Gemini فقط برای استخراج برنامه آموزشی و تولید سؤال فراخوانی می‌شود.
- حالت LM Studio به کلید گوگل نیاز ندارد و promptها فقط به سرور محلی ارسال می‌شوند.
- برای سؤال‌های مطمئن، خروجی JSON Schema درخواست می‌شود و fallback سازگاری وجود دارد.
- دسترسی منابع عمومی در برابر URLهای شبکه داخلی محافظت شده است.
- پیشرفت موضوعی، زمان پاسخ، امتیاز و تاریخچه بلوم دقیق ثبت می‌شوند.
""",
        )
    )
    if st.button(tr("Start", "شروع"), type="primary"):
        goto("language")

def language_page() -> None:
    st.title(tr("Choose language", "انتخاب زبان"))
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🇺🇸 English", width="stretch"):
            st.session_state.language = "english"
            goto("api")
    with col2:
        if st.button("🇮🇷 فارسی", width="stretch"):
            st.session_state.language = "persian"
            goto("api")
    if st.button(tr("Back", "برگشت")):
        goto("introduction")


def api_page() -> None:
    st.title(tr("Generation engine", "موتور تولید محتوا"))

    provider_labels = {
        "Gemini API": GENERATION_PROVIDER_GEMINI,
        "LM Studio — Local / محلی": GENERATION_PROVIDER_LMSTUDIO,
    }
    current_provider = st.session_state.get("provider", GENERATION_PROVIDER_GEMINI)
    current_label = next(
        (label for label, value in provider_labels.items() if value == current_provider),
        "Gemini API",
    )
    selected_label = st.radio(
        tr("Choose the generation engine", "موتور تولید را انتخاب کنید"),
        list(provider_labels),
        index=list(provider_labels).index(current_label),
        horizontal=True,
        key="generation_provider_widget",
    )
    provider = provider_labels[selected_label]
    st.session_state.provider = provider

    if provider == GENERATION_PROVIDER_GEMINI:
        st.info(
            tr(
                "Use a current Google AI Studio key. A 429 error with `limit: 0` means the key's project has no active quota for the selected model.",
                "از کلید فعلی Google AI Studio استفاده کنید. خطای 429 با `limit: 0` یعنی پروژه کلید برای مدل انتخابی سهمیه فعال ندارد.",
            )
        )
        with st.form("gemini_settings_form"):
            api_key = st.text_input(
                tr("API key", "کلید API"),
                value=st.session_state.api_key,
                type="password",
                placeholder="AIza...",
                key="gemini_api_key_input",
            )
            current_model = st.session_state.gemini_model
            index = SUPPORTED_MODELS.index(current_model) if current_model in SUPPORTED_MODELS else 0
            model = st.selectbox(
                tr("Text model", "مدل متنی"),
                SUPPORTED_MODELS,
                index=index,
                key="gemini_model_select",
            )
            submitted = st.form_submit_button(tr("Continue", "ادامه"), type="primary")
        if submitted:
            if not api_key.strip():
                st.error(tr("Enter an API key.", "کلید API را وارد کنید."))
            else:
                st.session_state.api_key = api_key.strip()
                st.session_state.gemini_model = model
                goto("setup")
    else:
        st.info(
            tr(
                "Start LM Studio's Local Server in the Developer tab. This Streamlit app and LM Studio must run on the same computer. The default server does not require a token.",
                "در تب Developer نرم‌افزار LM Studio، گزینه Local Server را روشن کنید. Streamlit و LM Studio باید روی همین رایانه اجرا شوند. سرور پیش‌فرض معمولاً Token نمی‌خواهد.",
            )
        )
        base_url = st.text_input(
            tr("LM Studio server URL", "نشانی سرور LM Studio"),
            value=st.session_state.lmstudio_base_url,
            placeholder="http://127.0.0.1:1234",
            key="lmstudio_url_input",
            help=tr(
                "Only loopback addresses are accepted for safety.",
                "برای امنیت فقط نشانی‌های loopback پذیرفته می‌شوند.",
            ),
        )
        token = st.text_input(
            tr("API token (optional)", "Token API (اختیاری)"),
            value=st.session_state.lmstudio_api_token,
            type="password",
            key="lmstudio_token_input",
            help=tr(
                "Needed only if Require Authentication is enabled in LM Studio Server Settings.",
                "فقط وقتی Require Authentication در تنظیمات سرور LM Studio روشن است لازم می‌شود.",
            ),
        )

        refresh_col, status_col = st.columns([1, 2])
        with refresh_col:
            refresh = st.button(
                tr("Check connection / Refresh models", "بررسی اتصال / دریافت مدل‌ها"),
                type="primary",
                width="stretch",
            )
        if refresh:
            try:
                normalized = normalize_lmstudio_base_url(base_url)
                models = list_lmstudio_models(normalized, token)
                st.session_state.lmstudio_base_url = normalized
                st.session_state.lmstudio_api_token = token.strip()
                st.session_state.lmstudio_models = models
                if models and st.session_state.lmstudio_model not in models:
                    st.session_state.lmstudio_model = models[0]
                if models:
                    st.success(
                        tr(
                            f"Connected. {len(models)} model(s) are visible.",
                            f"اتصال برقرار شد؛ {len(models)} مدل در دسترس است.",
                        )
                    )
                else:
                    st.warning(
                        tr(
                            "The server responded, but no model is visible. Download/load a Chat or Instruct model in LM Studio.",
                            "سرور پاسخ داد اما مدلی دیده نشد. در LM Studio یک مدل Chat یا Instruct دانلود/Load کنید.",
                        )
                    )
            except Exception as exc:
                show_engine_error(exc)

        models = list(st.session_state.get("lmstudio_models", []))
        if models:
            selected_index = (
                models.index(st.session_state.lmstudio_model)
                if st.session_state.lmstudio_model in models
                else 0
            )
            local_model = st.selectbox(
                tr("Local model", "مدل محلی"),
                models,
                index=selected_index,
                key="lmstudio_model_select",
                help=tr(
                    "Prefer a Chat/Instruct model with reliable JSON output.",
                    "مدل Chat/Instruct با خروجی JSON مطمئن انتخاب کنید.",
                ),
            )
        else:
            local_model = st.text_input(
                tr("Model identifier", "شناسه مدل"),
                value=st.session_state.lmstudio_model,
                key="lmstudio_manual_model_input",
                help=tr(
                    "You may enter the exact model id manually, but checking the connection first is recommended.",
                    "می‌توانید شناسه دقیق مدل را دستی وارد کنید، اما ابتدا بررسی اتصال پیشنهاد می‌شود.",
                ),
            )

        st.caption(
            tr(
                "Local requests do not set max_tokens and do not override the model's Thinking setting.",
                "درخواست‌های محلی max_tokens تعیین نمی‌کنند و تنظیم Thinking مدل را تغییر نمی‌دهند.",
            )
        )
        test_col, continue_col = st.columns(2)
        with test_col:
            test_model = st.button(
                tr("Test selected model", "تست تولید مدل انتخابی"),
                width="stretch",
            )
        if test_model:
            try:
                model_name = local_model.strip()
                if not model_name:
                    st.error(tr("Select or enter a local model.", "یک مدل محلی انتخاب یا وارد کنید."))
                else:
                    test_schema = {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"status": {"type": "string", "enum": ["ok"]}},
                        "required": ["status"],
                    }
                    result = generate_lmstudio_json(
                        prompt='Return exactly {"status":"ok"}.',
                        schema=test_schema,
                        schema_name="connection_test",
                        model=model_name,
                        base_url=base_url,
                        api_token=token,
                        temperature=0.0,
                    )
                    if result.get("status") == "ok":
                        st.success(tr("Local generation test passed.", "تست تولید محلی موفق بود."))
            except Exception as exc:
                show_engine_error(exc)

        with continue_col:
            continue_local = st.button(
                tr("Continue with LM Studio", "ادامه با LM Studio"),
                type="primary",
                width="stretch",
            )
        if continue_local:
            try:
                normalized = normalize_lmstudio_base_url(base_url)
                model_name = local_model.strip()
                if not model_name:
                    st.error(tr("Select or enter a local model.", "یک مدل محلی انتخاب یا وارد کنید."))
                else:
                    visible = list_lmstudio_models(normalized, token)
                    if visible and model_name not in visible:
                        st.warning(
                            tr(
                                "The entered model was not in /v1/models; LM Studio may still load it with JIT. Continuing with the entered identifier.",
                                "مدل واردشده در /v1/models نبود؛ ممکن است LM Studio آن را با JIT بارگذاری کند. با همین شناسه ادامه داده می‌شود.",
                            )
                        )
                    st.session_state.lmstudio_base_url = normalized
                    st.session_state.lmstudio_api_token = token.strip()
                    st.session_state.lmstudio_models = visible
                    st.session_state.lmstudio_model = model_name
                    goto("setup")
            except Exception as exc:
                show_engine_error(exc)

    if st.button(tr("Back", "برگشت")):
        goto("language")

def _add_resource_url(url: str) -> None:
    current = str(st.session_state.get("resource_urls_input", ""))
    urls = [line.strip() for line in current.splitlines() if line.strip()]
    if url not in urls:
        urls.append(url)
    st.session_state.resource_urls_input = "\n".join(urls)


def _resource_search_links(domain: str) -> None:
    """Discover direct, text-rich pages instead of linking to course catalogs."""
    if not domain:
        return

    has_persian = bool(re.search(r"[\u0600-\u06FF]", domain))
    if st.session_state.get("resource_search_domain") != domain:
        st.session_state.resource_search_domain = domain
        st.session_state.resource_search_query = "" if has_persian else domain
        st.session_state.resource_suggestions = []
        st.session_state.resource_suggestion_query = ""
        st.session_state.pop("resource_search_query_editor", None)

    with st.expander(tr("Find text-rich RAG resources", "یافتن منابع پرمتن مناسب RAG")):
        st.caption(
            tr(
                "The app searches the web, opens each candidate, extracts its readable text, and keeps only pages/PDFs that are sufficiently long and relevant. Course catalogs, enrollment pages, video sites, and thin landing pages are rejected.",
                "برنامه وب را جست‌وجو می‌کند، هر نتیجه را واقعاً باز و متن آن را استخراج می‌کند و فقط صفحه‌ها یا PDFهای طولانی و مرتبط را نگه می‌دارد. کاتالوگ دوره، صفحه ثبت‌نام، سایت ویدیویی و Landing Page کم‌متن حذف می‌شوند.",
            )
        )

        if has_persian and not st.session_state.resource_search_query:
            if st.button(
                tr("Translate topic for resource search", "تبدیل موضوع به عبارت جست‌وجوی انگلیسی"),
                key="translate_resource_search_query",
                type="primary",
            ):
                try:
                    with st.spinner(tr("Building English search terms...", "در حال ساخت عبارت انگلیسی...")):
                        expanded = expand_retrieval_query(
                            domain=domain,
                            subtopic="",
                            language="persian",
                            provider=st.session_state.provider,
                            model=active_model(),
                            api_key=(
                                st.session_state.api_key
                                if st.session_state.provider == GENERATION_PROVIDER_GEMINI
                                else None
                            ),
                            lmstudio_base_url=(
                                st.session_state.lmstudio_base_url
                                if st.session_state.provider == GENERATION_PROVIDER_LMSTUDIO
                                else None
                            ),
                            lmstudio_api_token=(
                                st.session_state.lmstudio_api_token
                                if st.session_state.provider == GENERATION_PROVIDER_LMSTUDIO
                                else None
                            ),
                        )
                    english_only = re.sub(r"[\u0600-\u06FF\u200c]+", " ", expanded)
                    english_only = re.sub(r"\s+", " ", english_only).strip(" ,;:-")
                    if not english_only:
                        raise ValueError("No English search terms were generated.")
                    st.session_state.resource_search_query = english_only
                    st.rerun()
                except Exception as exc:
                    show_engine_error(exc)
            st.info(
                tr(
                    "Translate the Persian topic first so English educational pages can be evaluated accurately.",
                    "ابتدا موضوع فارسی را به عبارت انگلیسی تبدیل کنید تا صفحه‌های آموزشی انگلیسی دقیق ارزیابی شوند.",
                )
            )
            return

        search_query = st.text_input(
            tr("English search query", "عبارت جست‌وجوی انگلیسی"),
            value=st.session_state.resource_search_query or domain,
            key="resource_search_query_editor",
            help=tr(
                "Use precise technical concepts. The query remains editable before discovery.",
                "مفاهیم فنی دقیق بنویسید. عبارت پیش از جست‌وجو قابل ویرایش است.",
            ),
        ).strip()
        st.session_state.resource_search_query = search_query
        if not search_query:
            return

        discovery_mode = st.radio(
            tr("Discovery depth", "عمق جست‌وجوی منابع"),
            options=["fast", "thorough"],
            index=0,
            horizontal=True,
            format_func=lambda value: tr(
                {
                    "fast": "Fast — validate the 10 best candidates",
                    "thorough": "Thorough — inspect a broader set",
                }[value],
                {
                    "fast": "سریع — بررسی ۱۰ گزینه برتر",
                    "thorough": "عمیق — بررسی گزینه‌های بیشتر",
                }[value],
            ),
            key="resource_discovery_mode",
            help=tr(
                "Fast mode runs search providers in parallel and downloads only the strongest candidates. Thorough mode takes longer.",
                "حالت سریع موتورهای جست‌وجو را موازی اجرا می‌کند و فقط بهترین گزینه‌ها را دانلود می‌کند. حالت عمیق زمان بیشتری می‌گیرد.",
            ),
        )

        if st.button(
            tr("Search and evaluate sources", "جست‌وجو و ارزیابی منابع"),
            type="primary",
            key="discover_text_rich_sources",
        ):
            try:
                spinner_text = (
                    tr(
                        "Quickly scanning and validating the strongest pages...",
                        "در حال اسکن سریع و اعتبارسنجی بهترین صفحه‌ها...",
                    )
                    if discovery_mode == "fast"
                    else tr(
                        "Running a broader search and content analysis...",
                        "در حال جست‌وجوی گسترده و تحلیل محتوای صفحه‌ها...",
                    )
                )
                with st.spinner(spinner_text):
                    suggestions = discover_text_rich_resources(
                        search_query,
                        max_results=(5 if discovery_mode == "fast" else 8),
                        max_candidates=(16 if discovery_mode == "fast" else 28),
                        mode=discovery_mode,
                    )
                st.session_state.resource_suggestions = suggestions
                st.session_state.resource_suggestion_query = f"{discovery_mode}:{search_query}"
            except Exception as exc:
                show_engine_error(exc)

        suggestion_key = f"{discovery_mode}:{search_query}"
        suggestions = (
            st.session_state.get("resource_suggestions", [])
            if st.session_state.get("resource_suggestion_query") == suggestion_key
            else []
        )
        if not suggestions:
            if st.session_state.get("resource_suggestion_query") == suggestion_key:
                st.warning(
                    tr(
                        "No candidate passed the text-length and relevance checks. Try a more specific English query or upload a PDF directly.",
                        "هیچ نتیجه‌ای از آزمون طول متن و ارتباط عبور نکرد. عبارت انگلیسی دقیق‌تری بنویسید یا PDF را مستقیم آپلود کنید.",
                    )
                )
            return

        st.success(
            tr(
                f"{len(suggestions)} directly readable sources passed RAG quality checks.",
                f"{len(suggestions)} منبع قابل‌خواندن از کنترل کیفیت RAG عبور کرد.",
            )
        )
        for index, item in enumerate(suggestions):
            title = item.get("title") or item.get("url")
            with st.container(border=True):
                st.markdown(f"**[{title}]({item['url']})**")
                st.caption(
                    tr(
                        f"{item.get('source', '')} · {item.get('word_count', 0):,} words · {item.get('text_chars', 0):,} text characters · relevance {item.get('relevance_score', 0):.0f}%",
                        f"{item.get('source', '')} · {item.get('word_count', 0):,} واژه · {item.get('text_chars', 0):,} نویسه متن · ارتباط {item.get('relevance_score', 0):.0f}٪",
                    )
                )
                if item.get("summary"):
                    st.write(item["summary"])
                if st.button(
                    tr("Add this source", "افزودن این منبع"),
                    key=f"add_rag_source_{index}_{abs(hash(item['url']))}",
                ):
                    _add_resource_url(item["url"])
                    st.success(tr("Added to the URL list.", "به فهرست URLها افزوده شد."))
                    st.rerun()


def setup_page() -> None:
    st.title(tr("Game setup", "تنظیم بازی"))

    domain_preview = st.text_input(
        tr("Learning domain", "موضوع یادگیری"),
        value=(st.session_state.setup or {}).get("domain", ""),
        placeholder=tr("Example: machine learning", "مثال: یادگیری ماشین"),
        key="domain_preview",
    )
    _resource_search_links(domain_preview)

    st.markdown(f"### 📄 {tr('Upload PDF resources', 'آپلود منابع PDF')}")
    uploaded_files = st.file_uploader(
        tr(
            "Drag and drop one or more searchable PDF files",
            "یک یا چند PDF دارای متن را بکشید و اینجا رها کنید",
        ),
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploaded_pdf_files_{st.session_state.pdf_uploader_nonce}",
        help=tr(
            "Text-based PDFs are read locally. Scanned/image-only PDFs require OCR first.",
            "PDFهای متنی به‌صورت محلی خوانده می‌شوند. PDF اسکن‌شده یا تصویری ابتدا به OCR نیاز دارد.",
        ),
    )

    newly_parsed_documents: list[dict[str, Any]] = []
    upload_errors: list[str] = []
    for uploaded in uploaded_files or []:
        try:
            document = cached_extract_uploaded_pdf(uploaded.name, uploaded.getvalue())
            newly_parsed_documents.append(document)
        except Exception as exc:
            upload_errors.append(f"{uploaded.name}: {exc}")

    if newly_parsed_documents and not upload_errors:
        st.session_state.uploaded_pdf_documents = newly_parsed_documents
    elif (
        not st.session_state.uploaded_pdf_documents
        and (st.session_state.setup or {}).get("uploaded_documents")
    ):
        st.session_state.uploaded_pdf_documents = (st.session_state.setup or {}).get(
            "uploaded_documents", []
        )
    uploaded_documents = list(st.session_state.uploaded_pdf_documents)

    if uploaded_documents:
        rows = [
            {
                tr("PDF", "PDF"): item["name"],
                tr("Pages", "صفحه"): item.get("pages", 0),
                tr("Readable characters", "نویسه قابل‌خواندن"): item.get("chars", 0),
                tr("Size (MB)", "حجم (MB)"): round(item.get("size_bytes", 0) / (1024 * 1024), 2),
            }
            for item in uploaded_documents
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
        st.success(
            tr(
                f"{len(uploaded_documents)} PDF files are ready for the local RAG corpus.",
                f"{len(uploaded_documents)} فایل PDF برای corpus محلی RAG آماده است.",
            )
        )
        if st.button(tr("Clear uploaded PDFs", "پاک‌کردن PDFهای آپلودشده"), key="clear_uploaded_pdfs"):
            st.session_state.uploaded_pdf_documents = []
            st.session_state.pdf_uploader_nonce += 1
            st.rerun()
    for error in upload_errors:
        st.error(error)

    if "resource_urls_input" not in st.session_state:
        st.session_state.resource_urls_input = "\n".join(
            (st.session_state.setup or {}).get("urls", [])
        )

    with st.form("setup_form"):
        urls_text = st.text_area(
            tr(
                "Text-rich educational pages or direct PDF URLs — one per line",
                "صفحه‌های آموزشی پرمتن یا URL مستقیم PDF — هر خط یک URL",
            ),
            height=170,
            placeholder="https://.../chapter-or-article\nhttps://.../lecture-notes.pdf",
            key="resource_urls_input",
            help=tr(
                "URLs are optional when at least one readable PDF is uploaded.",
                "اگر حداقل یک PDF قابل‌خواندن آپلود شده باشد، واردکردن URL اختیاری است.",
            ),
        )
        theme = st.text_input(
            tr("Narrative theme/style", "تم یا سبک روایت"),
            value=(st.session_state.setup or {}).get("title", "Academic adventure"),
            placeholder=tr("Serious, humorous, detective...", "جدی، طنز، کارآگاهی و ..."),
            key="narrative_theme_input",
        )
        age = st.number_input(
            tr("Learner age", "سن یادگیرنده"),
            min_value=6,
            max_value=100,
            value=int((st.session_state.setup or {}).get("age", 18)),
            key="learner_age_input",
        )
        mode_labels = {
            "Adaptive / تطبیقی — wrong answer lowers one Bloom level": "adaptive",
            "Progressive / پیشرونده — wrong answer keeps the same level": "progressive",
        }
        saved_mode = (st.session_state.setup or {}).get("game_mode", "adaptive")
        saved_mode_label = next(
            (label for label, value in mode_labels.items() if value == saved_mode),
            next(iter(mode_labels)),
        )
        selected_mode_label = st.radio(
            tr("Adaptation mode", "حالت تطبیق"),
            options=list(mode_labels),
            index=list(mode_labels).index(saved_mode_label),
            horizontal=False,
            key="game_mode_input",
        )
        game_mode = mode_labels[selected_mode_label]
        questions_per_topic = st.select_slider(
            tr("Questions per selected topic", "تعداد سؤال برای هر موضوع انتخابی"),
            options=[6, 10, 15, 20],
            value=int((st.session_state.setup or {}).get("questions_per_topic", 6)),
            key="questions_per_topic_input",
        )
        submitted = st.form_submit_button(
            tr("Analyze all resources", "تحلیل همه منابع"),
            type="primary",
        )

    if submitted:
        domain = domain_preview.strip()
        urls = list(
            dict.fromkeys(
                line.strip()
                for line in urls_text.splitlines()
                if line.strip()
            )
        )
        if not domain or not theme.strip():
            st.error(tr("Complete the topic and theme fields.", "موضوع و تم را کامل کنید."))
            return
        if not urls and not uploaded_documents:
            st.error(
                tr(
                    "Add at least one text-rich URL or upload one readable PDF.",
                    "حداقل یک URL پرمتن وارد کنید یا یک PDF قابل‌خواندن آپلود کنید.",
                )
            )
            return
        if upload_errors:
            st.error(
                tr(
                    "Remove or fix the unreadable PDF files before analysis.",
                    "پیش از تحلیل، PDFهای ناخوانا را حذف یا اصلاح کنید.",
                )
            )
            return

        invalid: list[str] = []
        for url in urls:
            valid, reason = validate_public_url(url, resolve_dns=False)
            if not valid:
                invalid.append(f"{url} — {reason}")
        if invalid:
            st.error(
                tr(
                    "Fix these URLs:\n" + "\n".join(f"- {item}" for item in invalid),
                    "این URLها را اصلاح کنید:\n" + "\n".join(f"- {item}" for item in invalid),
                )
            )
            return

        st.session_state.setup = {
            "domain": domain,
            "urls": urls,
            "uploaded_documents": uploaded_documents,
            "uploaded_pdf_names": [item["name"] for item in uploaded_documents],
            "title": theme.strip(),
            "age": int(age),
            "game_mode": game_mode,
            "questions_per_topic": int(questions_per_topic),
            "provider": st.session_state.provider,
            "model": active_model(),
            "lmstudio_base_url": (
                st.session_state.lmstudio_base_url
                if st.session_state.provider == GENERATION_PROVIDER_LMSTUDIO
                else None
            ),
        }
        st.session_state.candidate_subtopics = []
        try:
            with st.spinner(
                tr(
                    "Reading URLs and PDFs, then extracting source-grounded subtopics...",
                    "در حال خواندن URLها و PDFها و استخراج زیرموضوع‌های مبتنی بر منبع...",
                )
            ):
                subtopics = configured_extract_subtopics(
                    domain,
                    tuple(urls),
                    st.session_state.language,
                    uploaded_documents,
                )
            st.session_state.candidate_subtopics = subtopics
            goto("topics")
        except Exception as exc:
            show_engine_error(exc)
            return

    if st.button(tr("Back", "برگشت")):
        goto("api")


def topics_page() -> None:
    st.title(tr("Choose subtopics", "انتخاب زیرموضوع‌ها"))
    candidates = st.session_state.candidate_subtopics
    setup = st.session_state.setup
    if not candidates or not setup:
        goto("setup")
        return

    selected = st.multiselect(
        tr("Subtopics", "زیرموضوع‌ها"),
        options=candidates,
        default=candidates,
        key="selected_topics_widget",
    )
    total = len(selected) * setup["questions_per_topic"]
    st.info(
        tr(
            f"{len(selected)} topics × {setup['questions_per_topic']} questions = {total} questions",
            f"{len(selected)} موضوع × {setup['questions_per_topic']} سؤال = {total} سؤال",
        )
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(tr("Start game", "شروع بازی"), type="primary", width="stretch"):
            if not selected:
                st.warning(tr("Select at least one subtopic.", "حداقل یک زیرموضوع انتخاب کنید."))
            else:
                st.session_state.selected_subtopics = selected
                st.session_state.game = {
                    "language": st.session_state.language,
                    "provider": setup["provider"],
                    "domain": setup["domain"],
                    "urls": setup["urls"],
                    "uploaded_documents": setup.get("uploaded_documents", []),
                    "uploaded_pdf_names": setup.get("uploaded_pdf_names", []),
                    "title": setup["title"],
                    "age": setup["age"],
                    "model": setup["model"],
                    "lmstudio_base_url": setup.get("lmstudio_base_url"),
                    "game_mode": setup["game_mode"],
                    "questions_per_topic": setup["questions_per_topic"],
                    "subtopics": selected,
                    "levels": level_subtopics(selected),
                    "current_topic_idx": 0,
                    "bloom_idx": {topic: 0 for topic in selected},
                    "asked_by_topic": {topic: 0 for topic in selected},
                    "score": 0,
                    "correct_count": 0,
                    "total_answered": 0,
                    "history": {
                        topic: {
                            level: {"attempts": 0, "correct": 0}
                            for level in BLOOM_LEVELS
                        }
                        for topic in selected
                    },
                    "story": {topic: [] for topic in selected},
                    "response_times": [],
                }
                reset_question_state()
                goto("game")
    with col2:
        if st.button(tr("Edit setup", "ویرایش تنظیمات"), width="stretch"):
            goto("setup")


def _advance_finished_topics(game: dict[str, Any]) -> bool:
    qpt = game["questions_per_topic"]
    topics = game["subtopics"]
    while game["current_topic_idx"] < len(topics):
        topic = topics[game["current_topic_idx"]]
        if game["asked_by_topic"][topic] < qpt:
            return False
        game["current_topic_idx"] += 1
    return True


def _generate_current_question(game: dict[str, Any], topic: str, level: str) -> dict[str, Any]:
    asked = game["asked_by_topic"][topic]
    qpt = game["questions_per_topic"]
    position = "start" if asked == 0 else "end" if asked == qpt - 1 else "middle"
    provider = game.get("provider", GENERATION_PROVIDER_GEMINI)
    beat = generate_beat(
        topic,
        level,
        game["urls"],
        game["language"],
        game["title"],
        game["domain"],
        game["story"][topic],
        position,
        game.get("age"),
        uploaded_documents=game.get("uploaded_documents", []),
        provider=provider,
        api_key=(
            st.session_state.api_key
            if provider == GENERATION_PROVIDER_GEMINI
            else None
        ),
        model=game["model"],
        lmstudio_base_url=game.get("lmstudio_base_url"),
        lmstudio_api_token=(
            st.session_state.lmstudio_api_token
            if provider == GENERATION_PROVIDER_LMSTUDIO
            else None
        ),
    )
    options = list(beat["options"])
    random.SystemRandom().shuffle(options)
    beat["display_options"] = options
    return beat

def _bloom_progress(current_index: int) -> None:
    columns = st.columns(len(BLOOM_LEVELS))
    for index, level in enumerate(BLOOM_LEVELS):
        marker = "✅" if index < current_index else "🔵" if index == current_index else "⚪"
        with columns[index]:
            st.markdown(f"<div style='text-align:center'>{marker}<br><small>{level}</small></div>", unsafe_allow_html=True)


def game_page() -> None:
    game = st.session_state.game
    if not game:
        goto("setup")
        return
    if _advance_finished_topics(game):
        goto("feedback")
        return

    topic = game["subtopics"][game["current_topic_idx"]]
    bloom_index = int(game["bloom_idx"][topic])
    level = BLOOM_LEVELS[bloom_index]
    total_questions = len(game["subtopics"]) * game["questions_per_topic"]

    st.title(tr("Adaptive learning game", "بازی یادگیری تطبیقی"))
    st.caption(
        tr(
            f"Generator: {provider_label(game.get('provider'))} — {game.get('model', '')}",
            f"موتور تولید: {provider_label(game.get('provider'))} — {game.get('model', '')}",
        )
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric(tr("Score", "امتیاز"), game["score"])
    shown_question_number = game["total_answered"] if st.session_state.answer_result is not None else game["total_answered"] + 1
    metric_cols[1].metric(
        tr("Question", "سؤال"),
        f"{shown_question_number}/{total_questions}",
    )
    metric_cols[2].metric(
        tr("Topic", "موضوع"),
        f"{game['current_topic_idx'] + 1}/{len(game['subtopics'])}",
    )
    metric_cols[3].metric(tr("Bloom level", "سطح بلوم"), level)
    st.subheader(topic)
    _bloom_progress(bloom_index)

    if st.session_state.current_question is None:
        try:
            with st.spinner(tr("Generating one question...", "در حال تولید یک سؤال...")):
                st.session_state.current_question = _generate_current_question(game, topic, level)
                st.session_state.question_started_at = time.monotonic()
                st.session_state.answer_result = None
        except Exception as exc:
            show_engine_error(exc)
            col1, col2 = st.columns(2)
            with col1:
                if st.button(tr("Retry question", "تلاش مجدد برای سؤال"), type="primary", width="stretch"):
                    reset_question_state()
                    st.rerun()
            with col2:
                if st.button(tr("Return to setup", "بازگشت به تنظیمات"), width="stretch"):
                    goto("setup")
            return

    question = st.session_state.current_question
    with st.container(border=True):
        st.markdown(f"**📖 {tr('Story', 'داستان')}**")
        st.write(question["narrative"])
    st.markdown(f"### ❓ {question['question']}")

    answer = st.radio(
        tr("Choose one answer", "یک پاسخ را انتخاب کنید"),
        options=question["display_options"],
        index=None,
        disabled=st.session_state.answer_result is not None,
        key=f"answer_{game['total_answered']}_{topic}_{level}",
    )
    st.caption(
        tr(
            f"Time limit: {st.session_state.time_limit} seconds. It is checked when you submit.",
            f"مهلت پاسخ: {st.session_state.time_limit} ثانیه؛ هنگام ثبت پاسخ بررسی می‌شود.",
        )
    )

    if st.session_state.answer_result is None:
        if st.button(tr("Submit answer", "ثبت پاسخ"), type="primary"):
            if answer is None:
                st.warning(tr("Choose an answer first.", "ابتدا یک پاسخ انتخاب کنید."))
            else:
                elapsed = max(
                    0.0,
                    time.monotonic() - float(st.session_state.question_started_at),
                )
                timed_out = elapsed > st.session_state.time_limit
                is_correct = (answer == question["correct"]) and not timed_out

                history_row = game["history"][topic][level]
                history_row["attempts"] += 1
                history_row["correct"] += int(is_correct)
                game["total_answered"] += 1
                game["correct_count"] += int(is_correct)
                game["score"] += 10 if is_correct else 0
                game["response_times"].append(round(elapsed, 2))

                st.session_state.answer_result = {
                    "is_correct": is_correct,
                    "timed_out": timed_out,
                    "elapsed": elapsed,
                    "selected": answer,
                }
                st.rerun()
        return

    result = st.session_state.answer_result
    if result["timed_out"]:
        st.error(
            tr(
                f"Time expired after {result['elapsed']:.1f} seconds.",
                f"زمان پاسخ پس از {result['elapsed']:.1f} ثانیه تمام شد.",
            )
        )
    elif result["is_correct"]:
        st.success(tr("Correct!", "پاسخ درست است!"))
    else:
        st.error(tr("Incorrect.", "پاسخ نادرست است."))

    st.info(
        tr(
            f"Correct answer: {question['correct']}\n\nExplanation: {question['explanation']}",
            f"پاسخ درست: {question['correct']}\n\nتوضیح: {question['explanation']}",
        )
    )
    sources = question.get("_sources", [])
    if sources:
        with st.expander(tr("Sources used", "منابع استفاده‌شده")):
            for source in sources:
                if str(source).startswith("uploaded-pdf://"):
                    display_name = str(source).split("uploaded-pdf://", 1)[1].split("#", 1)[0]
                    st.write(f"📄 PDF: {display_name}")
                else:
                    st.write(source)

    if st.button(tr("Next question", "سؤال بعدی"), type="primary"):
        is_correct = bool(result["is_correct"])
        game["bloom_idx"][topic] = next_bloom_index(
            bloom_index,
            is_correct,
            game["game_mode"],
        )
        game["asked_by_topic"][topic] += 1
        game["story"][topic].append(question["narrative"])
        if game["asked_by_topic"][topic] >= game["questions_per_topic"]:
            game["current_topic_idx"] += 1
        reset_question_state()
        st.rerun()


def feedback_page() -> None:
    game = st.session_state.game
    if not game:
        goto("setup")
        return

    answered = int(game["total_answered"])
    correct = int(game["correct_count"])
    rate = correct / answered * 100 if answered else 0.0
    response_times = game.get("response_times", [])
    average_time = sum(response_times) / len(response_times) if response_times else 0.0

    st.title(tr("Learning results", "نتایج یادگیری"))
    columns = st.columns(4)
    columns[0].metric(tr("Final score", "امتیاز نهایی"), game["score"])
    columns[1].metric(tr("Answered", "پاسخ‌داده‌شده"), answered)
    columns[2].metric(tr("Success rate", "درصد موفقیت"), f"{rate:.1f}%")
    columns[3].metric(tr("Average time", "میانگین زمان"), f"{average_time:.1f}s")

    st.subheader(tr("Per-topic details", "جزئیات هر موضوع"))
    rows: list[dict[str, Any]] = []
    for topic in game["subtopics"]:
        topic_attempts = sum(
            game["history"][topic][level]["attempts"] for level in BLOOM_LEVELS
        )
        topic_correct = sum(
            game["history"][topic][level]["correct"] for level in BLOOM_LEVELS
        )
        rows.append(
            {
                tr("Topic", "موضوع"): topic,
                tr("Questions", "تعداد سؤال"): topic_attempts,
                tr("Correct", "درست"): topic_correct,
                tr("Success", "موفقیت"): f"{(topic_correct / topic_attempts * 100 if topic_attempts else 0):.1f}%",
                tr("Final Bloom level", "آخرین سطح بلوم"): BLOOM_LEVELS[game["bloom_idx"][topic]],
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

    export_state = {
        key: value
        for key, value in game.items()
        if key not in {"urls", "uploaded_documents"}
    }
    export_state["web_resource_count"] = len(game.get("urls", []))
    export_state["uploaded_pdf_count"] = len(game.get("uploaded_documents", []))
    export_state["uploaded_pdf_names"] = game.get("uploaded_pdf_names", [])
    export_state["resource_count"] = (
        export_state["web_resource_count"] + export_state["uploaded_pdf_count"]
    )
    export_state["success_rate"] = round(rate, 2)
    export_state["average_response_seconds"] = round(average_time, 2)
    st.download_button(
        tr("Download progress JSON", "دانلود پیشرفت به‌صورت JSON"),
        data=json.dumps(export_state, ensure_ascii=False, indent=2),
        file_name="adaptive_learning_progress.json",
        mime="application/json",
    )

    if st.button(tr("New game", "بازی جدید"), type="primary"):
        reset_game(keep_credentials=True)
        goto("setup")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    init_session_state()
    apply_css()
    sidebar_settings()

    pages = {
        "introduction": introduction_page,
        "language": language_page,
        "api": api_page,
        "setup": setup_page,
        "topics": topics_page,
        "game": game_page,
        "feedback": feedback_page,
    }
    page = pages.get(st.session_state.step, introduction_page)
    page()


if __name__ == "__main__":
    main()
