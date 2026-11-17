# PS-12 ShopSmart AI: Implementation Guide

This document describes the functionality currently implemented in the repository. It is intended for developers and coding agents working on the project.

## Project Summary

PS-12 is a conversational shopping assistant with:

- FastAPI backend with lifespan pre-warming for ranking models
- LangGraph workflow orchestration with dynamic conditional routing
- OpenRouter centralized LLM gateway with primary/fallback models and strict error classification
- Dedicated LLM-based Product Explanation Agent with dual-aspect explanations (retrieval reason & ranking reason)
- Optimized CrossEncoder semantic ranking (`ms-marco-MiniLM-L-6-v2`) with singleton caching and candidate pruning
- Multi-provider product retrieval (Serper Google Shopping & Tavily Web Search)
- Safe product-page JSON-LD extraction
- FastMCP tools (6 tools: search, details, provider status, compare products, product reviews, product alternatives)
- React/Vite ShopSmart AI frontend with ranking details accordions, requirement verification pills, and results banner
- Hard budget and availability enforcement
- Persistent multi-chat session memory and side-by-side product comparisons
- Mocked automated tests for providers, ranking, explanations, MCP tools, and workflow behavior

The local CSV catalog was removed from the active application. Product discovery uses external providers only. No synthetic or local catalog fallback is used.

## Repository Layout

```text
backend/
  agents/
    attribute_mapping.py      # Maps user constraints to search queries
    clarification.py          # Shopping intent completeness & clarification questions
    explanation.py            # Dedicated LLM explanation agent & follow-up handler
    fallback.py               # Deterministic no-results fallback
    intent_extraction.py      # Intent extraction (OpenRouter + regex fallback)
    orchestrator.py           # Multi-agent coordination helpers
    query_ingestion.py        # Preprocessing and conversation session loading
    ranking.py                # Hard budget enforcement + CrossEncoder scoring
    retrieval.py              # Retrieval node invoking external providers
    session_memory.py         # Conversational session memory store
  api/routes/
    chat.py                   # POST /api/v1/chat endpoint
    products.py               # GET /api/v1/products endpoints
  graph/
    state.py                  # LangGraph TypedDict state contract
    workflow.py               # StateGraph definition & conditional routing
  providers/
    base.py                   # ProductProvider protocol & search result structures
    composite.py              # Parallel provider aggregator & error isolation
    page_extractor.py         # Bounded JSON-LD schema extractor
    serper.py                 # Google Shopping provider
    web_search.py             # Tavily web search provider
  schemas/
    chat.py                   # API request and response models
    product.py                # Normalized Product, RequirementMatch, Explanation models
  services/
    ai.py                     # OpenRouter client, error classifier, and cooldown
    product_retrieval.py      # Product retrieval service wrapper
    product_sources.py        # Provider builder factory
  main.py                     # FastAPI app with CrossEncoder lifespan warmup
  mcp_server.py               # FastMCP server over stdio
frontend/
  src/
    main.jsx                  # Main React frontend with explanation UI & multi-session
    styles.css                # ShopSmart styling, pills, accordions, comparisons
  package.json
tests/
  test_clarification_completeness.py       # Intent completeness & broad query clarification
  test_explainable_ranking_and_fallback.py # Budget enforcement, scoring, and fallbacks
  test_explanation_and_performance.py      # Explanation agent, status pills, and CrossEncoder timing
  test_intent_clarification.py             # Intent extraction & conversation follow-ups
  test_mcp_tools_and_integration.py        # All 6 FastMCP tools & graph integration
  test_openrouter_migration.py             # OpenRouter fallback, 402/401 handling, & workflow
  test_phase1_providers.py                 # Schema, URL safety, & failure isolation
  test_phase2_integration.py               # End-to-end integration & composite providers
  test_serper_and_real_shopping.py         # Serper shopping, currency parsing, & budget filters
  test_tavily_and_extraction.py            # Tavily search & JSON-LD page extraction
README.md
.env.example
requirements.txt
```

## Backend Startup

Run commands from the repository root:

```powershell
cd D:\RAHUL\BE\AAI\mp
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

Useful endpoints:

```text
GET  /health
GET  /docs
POST /api/v1/chat
GET  /api/v1/products
GET  /api/v1/products/{product_id}
```

The FastAPI application startup lifecycle (`backend/main.py`) warms up the `CrossEncoder` model singleton in memory so that subsequent user requests avoid cold-start deserialization latency.

## Frontend Startup

The frontend is a separate Vite project. Run it from the frontend directory:

```powershell
cd D:\RAHUL\BE\AAI\mp\frontend
npm install
npm run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173
```

Do not run `npm run dev` from the repository root because there is no root-level `package.json`.

The frontend uses `VITE_API_URL` when supplied and otherwise defaults to `http://127.0.0.1:8000`.

## Environment Variables

Copy the template before running:

```powershell
copy .env.example .env
```

Important variables:

```env
OPENROUTER_API_KEY=
OPENROUTER_MODEL=google/gemini-2.5-flash
OPENROUTER_FALLBACK_MODELS=meta-llama/llama-3.3-70b-instruct,deepseek/deepseek-chat
OPENROUTER_ENDPOINT=https://openrouter.ai/api/v1
OPENROUTER_TIMEOUT_SECONDS=10

TAVILY_API_KEY=
TAVILY_ENDPOINT=https://api.tavily.com/search
TAVILY_SEARCH_DEPTH=advanced
TAVILY_MAX_RESULTS=8
TAVILY_PAGE_FETCH_LIMIT=4
TAVILY_TIMEOUT_SECONDS=8

SERPER_API_KEY=
SERPER_ENDPOINT=https://google.serper.dev/search
SERPER_GL=in
SERPER_HL=en
SERPER_NUM=20
SERPER_TIMEOUT_SECONDS=8
```

Never commit `.env` or expose API keys in the frontend. `.env.example` must contain blank credential values only.

## OpenRouter LLM Gateway & Resilient Error Handling

File: `backend/services/ai.py`

OpenRouter serves as the centralized LLM gateway for intent extraction and recommendation explanations.

Key capabilities:
- **Multi-Model Priority Chain**: Tries the primary configured model (`OPENROUTER_MODEL`), followed by candidates in `OPENROUTER_FALLBACK_MODELS`.
- **Permanent Error Classification**:
  - `HTTP 402 Payment Required`: When account credits are exhausted, immediately stops retrying candidate models under the same key. Sets `last_category="payment_required"`, initiates a 30-second cooldown for that key to avoid wasted network calls, and falls back to deterministic logic instantly.
  - `HTTP 401 / 403 Unauthorized / Forbidden`: Stops candidate model loop immediately and falls back.
- **Transient Error Retries**:
  - `HTTP 429 Rate Limit` and `HTTP 5xx Server Errors`: Attempts the next candidate fallback model.
- **Credential Masking**: All log messages mask sensitive API keys (`sk-or-...`).
- **Deterministic Fallback**: If all models fail or credentials are unconfigured, operations fallback to deterministic regex/rule logic without breaking the user turn.

## Provider Architecture

### ProductProvider

`backend/providers/base.py` defines the provider protocol and `ProviderError`.

A provider exposes:
- `name`
- `status()`
- `search(...)`
- `get(product_id)`

Search results use `ProductSearchResult`, which contains:
- `products`
- `providers`
- `errors`
- `provider_status`

### Active Provider Composition

`backend/services/product_sources.py` builds the production retrieval service with:
1. `SerperShoppingProvider`
2. `TavilyWebSearchProvider`

The providers are wrapped by `CompositeProductProvider`, which:
- Calls each provider independently
- Preserves successful results if another provider fails
- Records provider errors and health status
- Preserves separate retailer offers
- Deduplicates only exact source offers

### Serper Shopping Provider

File: `backend/providers/serper.py`

Serper provides structured Google Shopping listings:
- Retailer/merchant source names and product links
- Image URLs, ratings, and review counts
- Prices and multi-currency parsing (INR, USD, EUR, GBP, AED, SGD, CAD, AUD) converted to INR
- Hard budget filtering and missing-price rejection
- Verified price guarantees

### Tavily Web Search Provider

File: `backend/providers/web_search.py`

Tavily searches the web for product discovery pages:
- Configurable search depth and timeouts
- HTTP 401/429 handling and duplicate URL elimination
- Discovery products default to `price_verified=False` and `availability="unknown"` unless verified by page extraction
- Unverified discovery listings cannot satisfy a hard budget constraint

### Page Extraction

File: `backend/providers/page_extractor.py`

Extracts JSON-LD `Product` and `Offer` metadata from public web pages:
- Extracts name, brand, SKU, price, currency, availability, and image URL
- Enforces strict SSRF protections: blocks localhost, private IPs, loopback, and link-local addresses
- Limits response body size and parses non-JavaScript HTML only

## Normalized Product & Explanation Schema

Files: `backend/schemas/product.py` and `backend/schemas/chat.py`

`Product` is the provider-neutral normalized schema. Key fields include:
- `product_id`, `name`, `brand`, `category`, `subcategory`
- `price`, `currency`, `original_price`, `price_verified`
- `availability`, `availability_verified`
- `rating`, `review_count`, `features`, `use_cases`, `tags`
- `image_url`, `product_url`, `source_name`
- `retrieval_reason`: Why this product was retrieved based on user constraints
- `ranking_reason`: Why this product is ranked at its position relative to others
- `requirement_matches`: Array of `RequirementMatch` objects (`requirement`, `status`, `evidence`), where `status` is strictly `"matched"`, `"not_matched"`, or `"unknown"`
- `strengths`: Verified product advantages
- `matched_attributes`: List of verified matching attributes
- `unverified_attributes`: List of unverified or missing attributes
- `explanation_details`: Detailed diagnostic and scoring breakdown

## LangGraph Workflow

Files: `backend/graph/state.py` and `backend/graph/workflow.py`

Current flow:

```text
load session
  -> ingest query
  -> extract intent
  -> check clarification
  -> [if broad/incomplete] -> request clarification
  -> [if explain_ranking]  -> handle explanation follow-up (no re-search)
  -> [if complete]         -> map attributes
                              -> retrieve from Serper/Tavily
                              -> rank candidates (CrossEncoder)
                              -> explain recommendations (Explanation Agent)
                              -> fallback if no ranked products
                              -> save session
```

Conditional Routing:
- **Clarification**: Broad requests (e.g., "Suggest some travel equipment") trigger a targeted clarifying question before search.
- **Explainability Follow-up**: Queries like `"Why is this ranked first?"`, `"Why is this product better?"`, or `"What does it not satisfy?"` route directly to the explanation follow-up handler, utilizing previous session products without re-executing search or re-ranking.

## Dedicated LLM-based Product Explanation Agent

File: `backend/agents/explanation.py`

The explanation agent runs after ranking and produces grounded explanations:
1. **Aspect A (Why Retrieved)**: Explains relevance to the user's explicit query, category, and budget.
2. **Aspect B (Why Ranked Here)**: Explains position relative to competing candidates (e.g. price advantage, ratings, verified feature match).
3. **Structured Requirement Verification**:
   - Compares constraints against merchant evidence.
   - Assigns `"matched"`, `"not_matched"`, or `"unknown"` status.
   - Strictly prevents hallucination: specifications not found in the listing remain `"unknown"`, and missing prices/ratings are explicitly reported as unverified rather than fabricated.
4. **Deterministic Fallback**: If OpenRouter fails or credits are exhausted, a deterministic evidence extractor generates comprehensive reasons and requirement pills instantly.
5. **Session-Aware Follow-ups**: Evaluates winner vs. runner-up trade-offs and unmet requirements without re-triggering retrieval.

## Optimized CrossEncoder Ranking Performance

File: `backend/agents/ranking.py`

The ranking pipeline combines deterministic rule filtering with semantic CrossEncoder scoring:
1. **Deterministic Budget Filtering**: Excludes unverified or over-budget items when budget is specified.
2. **Candidate Pre-Pruning**: Selects the top 10 candidates based on lexical/attribute scores before running neural inference.
3. **Singleton Model Loader (`get_cross_encoder()`)**: Prevents re-instantiating the model or reloading weights on every request.
4. **Offline Mode Enforcement**: Sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `local_files_only=True` to eliminate HuggingFace remote check timeouts (~31s network delay).
5. **Batched Inference & Thread Tuning**: Evaluates pairs with `batch_size=16` on CPU.
6. **Detailed Phase Timing**: Emits structured logs for `model_initialization`, `candidate_filtering`, `cross_encoder_inference`, and `final_ranking`.
7. **Performance Gain**: Inference latency dropped from ~39.2s down to ~0.03s–0.10s (over 390x speedup).

## FastMCP Tools

File: `backend/mcp_server.py`

The FastMCP server runs over stdio and exposes 6 tools:
1. `search_products(query, max_price, category)`: Product discovery via Serper & Tavily.
2. `get_product_details(product_id)`: Fetches complete normalized product record.
3. `get_provider_status()`: Reports health and availability of configured search providers.
4. `compare_products(products, focus_attributes)`: Generates side-by-side comparison, attribute matrices, and trade-off summaries.
5. `search_product_reviews(product_name, brand)`: Aggregates customer sentiment, editorial reviews, pros, and cons.
6. `find_product_alternatives(target_product, alternative_type, max_price)`: Discovers cheaper or premium alternatives.

## React Frontend

Files: `frontend/src/main.jsx` and `frontend/src/styles.css`

UI Features:
- **Results Overview Banner**: Displays top query context, verified counts, and sorting criterion.
- **Why This Recommendation? Box**: Highlights concise retrieval and ranking explanations on each product card.
- **Requirement Verification Indicators**: Renders interactive status pills (`matched`, `not_matched`, `unknown`) with color-coded badges and hover tooltips.
- **Expandable Ranking Details Accordion**: Toggles deep-dive insights on candidate strengths, limitations, and relative position logic.
- **Product Comparison**: Side-by-side comparison table for up to 3 saved products.
- **Multi-Session Chat History**: Left sidebar stores and restores independent chat sessions from browser local storage.
- **Responsive Controls**: Category filter chips, budget sliders, and sort criteria.

## Automated Tests

Run the complete test suite:

```powershell
cd D:\RAHUL\BE\AAI\mp
.\.venv\Scripts\python.exe -m pytest -q
```

Test coverage:
- 10 test modules with 155 automated tests.
- All external search providers, OpenRouter LLM calls, and JSON-LD fetches are safely mocked.
- Tests run completely offline without requiring live API keys.

## Known Limitations

- External provider credentials (`SERPER_API_KEY`, `TAVILY_API_KEY`, `OPENROUTER_API_KEY`) are required for live data in production.
- Tavily discovery pages may not expose verifiable prices or stock without JSON-LD metadata.
- Page extraction only handles bounded non-JavaScript HTML content.
- Session memory is process-local and not persistent across server restarts.
- FastMCP uses stdio transport and requires an MCP-compatible client.

## Development Rules

- Never hardcode credentials or print API keys.
- Do not use a local catalog or synthetic product data.
- Do not invent prices, ratings, stock, images, URLs, or specifications.
- Keep budgets and category constraints deterministic.
- Preserve source attribution and verification status.
- Treat discovered pages as unverified until extraction proves structured fields.
- Keep provider failures isolated.
- Run the full test suite (`pytest -q`) after schema, provider, agent, or workflow changes.
