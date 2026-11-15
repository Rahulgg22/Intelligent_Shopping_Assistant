# PS-12 ShopSmart AI: Implementation Guide

This document describes the functionality currently implemented in the repository. It is intended for developers and coding agents working on the project.

## Project Summary

PS-12 is a conversational shopping assistant with:

- FastAPI backend
- LangGraph workflow orchestration
- Gemini-assisted intent extraction and explanations
- Multi-provider product retrieval
- Serper Google Shopping integration
- Tavily web-search integration
- Safe product-page JSON-LD extraction
- MCP v2 tools
- React/Vite ShopSmart AI frontend
- Hard budget and availability enforcement
- Saved products and comparison UI state
- Mocked automated tests for providers and workflow behavior

The local CSV catalog was removed from the active application. Product discovery now uses external providers only. The CSV provider implementation and catalog-dependent tests are no longer part of the production retrieval path.

## Repository Layout

```text
backend/
  agents/
    attribute_mapping.py
    clarification.py
    explanation.py
    fallback.py
    intent_extraction.py
    orchestrator.py
    query_ingestion.py
    ranking.py
    retrieval.py
    session_memory.py
  api/routes/
    chat.py
    products.py
  graph/
    state.py
    workflow.py
  providers/
    base.py
    composite.py
    page_extractor.py
    serper.py
    web_search.py
  schemas/
    chat.py
    product.py
  services/
    ai.py
    product_sources.py
    product_retrieval.py
  main.py
  mcp_server.py
frontend/
  src/main.jsx
  src/styles.css
  package.json
scripts/
tests/
  test_phase1_providers.py
  test_phase2_integration.py
  test_serper_and_real_shopping.py
  test_tavily_and_extraction.py
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

The product endpoint uses the same external retrieval service as the chat workflow. It does not read a local catalog.

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
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-2.0-flash

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

`backend/services/catalog.py` builds the production retrieval service with:

1. `SerperShoppingProvider`
2. `TavilyWebSearchProvider`

The providers are wrapped by `CompositeProductProvider`, which:

- Calls each provider independently
- Preserves successful results if another provider fails
- Records provider errors
- Records provider states
- Preserves separate retailer offers
- Deduplicates only exact source offers

### Serper Shopping Provider

File: `backend/providers/serper.py`

Serper is the structured shopping-results provider. It is the preferred source for current shopping listings when `SERPER_API_KEY` is configured.

It supports:

- Google Shopping-style results
- Retailer/source names
- Product links
- Images
- Prices
- Currency parsing
- INR normalization
- Ratings and review counts when returned
- Delivery information
- Hard budget filtering
- Missing-price rejection
- Rate-limit and authentication errors

Supported currency parsing includes INR, USD, EUR, GBP, AED, SGD, CAD, and AUD. Foreign prices are converted to INR using the configured benchmark conversion table in the provider.

Serper results with verified prices can satisfy a budget. Results without prices or above the budget are excluded from compliant recommendations.

### Tavily Web Search Provider

File: `backend/providers/web_search.py`

Tavily is the discovery provider. It searches for relevant product pages and returns discovery records. Tavily alone does not guarantee current price, stock, rating, or product metadata.

It supports:

- Configurable search depth
- Configurable result limits
- Timeouts
- Duplicate URL removal
- Missing-key handling
- HTTP 401 and 429 handling
- Category relevance filtering
- Optional page extraction
- Provider status reporting

Tavily discovery products normally have:

- `price=None`
- `price_verified=False`
- `availability="unknown"`

Therefore, Tavily-only results do not satisfy a hard budget unless page extraction verifies an INR price.

### Page Extraction

File: `backend/providers/page_extractor.py`

The extractor uses permitted, bounded HTTP fetches and parses JSON-LD Product/Offer metadata.

It can extract:

- Product name
- Brand
- SKU
- Description
- Price
- Currency
- Availability
- Image URL
- Canonical/source URL
- Retrieval timestamp
- Field-level provenance

Security controls:

- HTTP/HTTPS only
- Localhost blocked
- Private, loopback, link-local, and reserved IPs blocked
- Redirect targets revalidated
- HTML/XHTML content types only
- Response body bounded before parsing
- No JavaScript execution
- No authentication bypass
- No CAPTCHA or bot-protection bypass

If extraction fails, the original Tavily discovery record can remain available as unverified discovery data, but it cannot satisfy a hard price constraint.

## Normalized Product Schema

File: `backend/schemas/product.py`

`Product` is the provider-neutral data contract. Important fields include:

- `product_id`
- `name`
- `brand`
- `category`
- `subcategory`
- `price`
- `currency`
- `original_price`
- `description`
- `features`
- `use_cases`
- `tags`
- `rating`
- `review_count`
- `image_url`
- `product_url`
- `source_name`
- `source_product_id`
- `availability`
- `price_verified`
- `availability_verified`
- `retrieved_at`
- `match_score`
- `match_reasons`
- `model_number`
- `data_provenance`

Unknown values remain `None`, empty, or `unknown`. Missing values must not be changed into fabricated zero values.

URL validation rejects unsupported schemes and unsafe internal destinations.

## LangGraph Workflow

Files:

- `backend/graph/state.py`
- `backend/graph/workflow.py`

Current flow:

```text
load session
  -> ingest query
  -> extract intent
  -> check clarification
  -> map attributes
  -> retrieve from Serper/Tavily
  -> rank candidates
  -> explain recommendations
  -> fallback if no ranked products
  -> save session
```

The retrieval node calls `build_product_service()` and places these fields into graph state:

- `candidates`
- `provider_status`
- `retrieval_errors`

The ranking node applies deterministic budget enforcement before scoring. Unknown prices are excluded when a budget is specified. CrossEncoder scoring is optional and has a deterministic fallback.

## Intent and Conversation Handling

Intent extraction supports:

- Product categories and subcategories
- Budget values
- Indian currency expressions
- Activities/use cases
- Preferences
- Recipients
- Short follow-up messages

Independent queries reset stale category context. Follow-up phrases such as “make it cheaper”, “different brand”, or “same category” preserve relevant prior constraints.

Clarification is only requested when an essential category or product type is missing. Gift queries receive gift-specific clarification rather than outdoor-specific wording.

Session memory is currently process-local in `backend/agents/session_memory.py`. It supports conversational behavior but is not durable across backend restarts.

## Fallback Behavior

File: `backend/agents/fallback.py`

The fallback does not use a local catalog. If no verified external products remain after filtering, it returns:

- `action="message"`
- Empty `products`
- A query-specific no-results explanation

It never fills results with unrelated products or silently relaxes a hard budget.

## FastAPI Response Contract

Files:

- `backend/api/routes/chat.py`
- `backend/schemas/chat.py`

Chat responses include:

- `session_id`
- `action`
- `message`
- `products`
- `clarifying_question`
- `active_constraints`
- `provider_status`
- `retrieval_errors`
- `metadata`

Each product can include source, price verification, availability verification, links, ratings, images, and provenance-related fields.

## MCP Server

File: `backend/mcp_server.py`

The MCP server uses MCP SDK v2 and stdio transport:

```powershell
cd D:\RAHUL\BE\AAI\mp
.\.venv\Scripts\python.exe -m backend.mcp_server
```

Available tools:

- `search_products`
- `get_product_details`
- `get_provider_status`

MCP calls the same `build_product_service()` used by FastAPI. It is not a separate fake data path. It does not expose a browser URL or TCP port.

The MCP server is a separate interface. The browser frontend calls FastAPI directly; it does not automatically call MCP.

## React Frontend

Files:

- `frontend/src/main.jsx`
- `frontend/src/styles.css`

Implemented UI features:

- ShopSmart AI three-area layout
- Conversation panel
- Example prompts
- Loading state
- Error state
- Provider status display
- Product cards
- Price and availability verification labels
- External source links
- Responsive filters
- Category filtering
- Budget filtering
- Sorting
- Saved products in browser local storage
- Comparison of up to three products
- Persistent browser session ID
- Multiple persistent chat sessions in browser local storage
- New Chat creates a fresh backend session without deleting previous chats
- Sidebar chat history with reopen behavior
- My Searches archive view
- Migration of the previous single `shopsmart-history` storage into the first chat session

The frontend does not contain hardcoded product result arrays. It uses FastAPI responses.

## Tests

Run the complete suite:

```powershell
cd D:\RAHUL\BE\AAI\mp
.\.venv\Scripts\python.exe -m pytest -q
```

The current suite covers:

- Normalized product fields
- Safe URL validation
- Provider failure isolation
- Multiple retailer offers
- Tavily missing credentials
- Tavily mocked search results
- Tavily authentication and rate-limit failures
- JSON-LD page extraction
- Unsafe page URLs
- Serper credential/error behavior
- Serper currency parsing
- Serper laptop, suitcase, and headphone searches
- Budget enforcement
- Composite provider behavior
- LangGraph retrieval
- MCP tools
- Empty-result behavior
- Session and follow-up behavior

The latest verified total before this document was created was `14 passed` after the external-only/Tavily changes, and the repository also contains Serper-specific tests that should be run after any provider changes.

## Known Limitations

- External provider credentials are required for live results.
- Serper and Tavily may return different levels of product detail.
- Tavily discovery pages may not expose verifiable prices or stock.
- Page extraction only handles bounded non-JavaScript page content and structured metadata.
- No retailer-specific API is currently implemented.
- Session memory is process-local and not durable.
- MCP uses stdio and requires an MCP-compatible client.
- Product comparison and saved items are browser-local features.
- The frontend can display no-results fallbacks when external sources return no verified products.
- Search result quality depends on the configured provider and source-page accessibility.

## Development Rules

- Never hardcode provider credentials.
- Never print API keys.
- Do not use the removed local catalog as a hidden fallback.
- Do not invent prices, ratings, stock, images, URLs, or specifications.
- Keep budgets and category constraints deterministic.
- Preserve source attribution and verification status.
- Treat discovered pages as unverified until extraction supports a field.
- Keep provider failures isolated.
- Add mocked tests for external provider behavior; tests must not require live credentials.
- Run the full test suite after provider, schema, graph, or response changes.
