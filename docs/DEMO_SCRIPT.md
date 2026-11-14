# Prism Demo Script

## Before recording

1. Open **Terminal**.
2. Copy and paste this command exactly:

```bash
cd "/Users/chandramukhi/Documents/Codex/2026-11-10/referenced-chatgpt-conversation-this-is-an/prism-ai-gateway" && bash scripts/run_local.sh
```

3. Wait until Terminal shows:

```text
Prism ready — providers: openrouter
```

4. Do not close Terminal. Leave it behind your browser window.
5. Open this in your browser:

```text
http://localhost:8080/console/chat.html
```

## Recording: what to do and say

### 1. Introduce the project (15 seconds)

Say:

> This is Prism, an AI gateway. Instead of an application calling an AI provider directly, it sends requests through Prism. Prism checks the API key, chooses a model, checks the cache, sends the request to the provider, and records usage.

### 2. Show a real answer (30 seconds)

On the chat page:

1. Leave **Search team — cache enabled** selected.
2. Leave **Fast** selected.
3. Replace the question with:

```text
Explain what an API gateway does in two simple sentences.
```

4. Click **Send message**.

Say:

> This request is going through Prism. The answer below comes from the configured AI provider. The labels show which provider served it, whether it came from cache, and the tracked gateway cost.

### 3. Show the cache (20 seconds)

1. Click **Send message** again without changing the question.
2. Look for **Cache: hit** below the answer.

Say:

> I asked the same question again. Prism recognized a matching cached response, so it returned the answer without sending another provider request. This reduces latency and cost.

### 4. Show the monitoring dashboard (30 seconds)

1. Click **Ops console** at the top right.
2. Scroll to **Recent Requests**.

Say:

> The operations console records each request. It shows the selected model, provider, cache result, tokens, cost, and response time. The top cards summarize total requests, spending, tokens, and cache performance.

### 5. Show the architecture document (30 seconds)

Open a second Terminal window and run:

```bash
cd "/Users/chandramukhi/Documents/Codex/2026-11-10/referenced-chatgpt-conversation-this-is-an/prism-ai-gateway" && open -a TextEdit docs/ARCHITECTURE.md
```

Show the first diagram and say:

> The architecture has a client layer, Prism as the gateway, access control for virtual keys and limits, routing and caching, provider adapters, and a database for usage and logs. This separation lets Prism support different providers without changing the client API.

### 6. Finish (10 seconds)

Say:

> Prism demonstrates a practical AI gateway with OpenAI-compatible requests, controlled access, intelligent routing, semantic caching, usage metering, and an operations console.

## If OpenRouter is unavailable

Use the key-free local demo instead. In Terminal, press `Control-C` to stop Prism, then run:

```bash
PRISM_GATEWAY_CONFIG_PATH=data/gateway_config.sample.json bash scripts/run_local.sh
```

The same chat page and dashboard will work. The response metadata will mention `alpha` or `beta`; explain that they are built-in OpenAI-compatible mock providers used for offline verification.

## After recording

Press `Control-C` in the Terminal that is running Prism.

Never show your `.env` file or API key in the recording.
