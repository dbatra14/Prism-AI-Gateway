# Prism — LLM Gateway & Semantic Cache

One door for all AI traffic. Applications call Prism with a virtual key; Prism
decides which model should answer, whether it needs to call a provider at all,
enforces what each team is allowed to spend, and records where every cent went.

```
   your apps ──▶ ┌─────────── PRISM ───────────┐ ──▶ alpha  (mock provider)
                 │ auth · limits · budgets     │ ──▶ beta   (mock provider)
                 │ smart routing · cache       │      …or OpenAI, Anthropic, …
                 │ retries · failover          │
                 │ metering · request log      │
                 └─────────────────────────────┘
```

Two decisions are made by machine judgement **inside the request path**, which
is what makes this more than a proxy:

- **Smart routing** — the gateway reads each prompt and judges how hard it is,
  sending easy work to a cheap model and hard work to a strong one. The strong
  model costs 20× more, so this is a real money decision made per request.
- **Semantic cache** — repeated questions are matched by *meaning*, not by text.
  *"If I bought an annual plan, can I get my money back?"* and *"What is the
  refund policy for annual plans?"* share almost no words but are the same
  question, and the second one is answered for free.

**Verified results** (reproduce with `python scripts/verify.py` — see
[VERIFICATION.md](VERIFICATION.md)):

| | |
|---|---|
| Provided smoke test | 18 passed, 0 warnings, 0 failed |
| Routing accuracy | **20/20** vs 11/20 for a length-only baseline |
| …on trap cases only | **8/8** vs 1/8 for the baseline |
| Over-admission under a concurrent burst | **10 accepted, limit 10** |
| Accounting under concurrency | client-observed totals reconcile with the usage API |
| Automated tests | **138 passing** |

---

## Contents

- [Quick start](#quick-start) · [The five-minute demo](#the-five-minute-demo) · [Submission guide](docs/SUBMISSION.md)
- [How a request flows](#how-a-request-flows) · [Reading the code](#reading-the-code)
- [API reference](#api-reference)
- [Design decisions](#design-decisions): [rate limiting](#1-rate-limiting-without-over-admission) ·
  [budgets](#2-budgets-and-the-overshoot-we-accept) · [routing](#3-smart-routing-length-is-a-trap) ·
  [cache scoping](#4-cache-scoping-and-tenant-isolation) · [failover](#5-retries-failover-and-the-streaming-rule)
- [Testing](#testing) · [Known limitations](#known-limitations)

---

## Quick start

Requires Python 3.11+. The first run downloads a ~130MB embedding model.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

Prism needs at least two providers to route between. The provided mock providers
are OpenAI-compatible and support failure injection, so the whole demo runs
offline and free. **Use three terminals:**

```bash
python scripts/mock_provider.py --port 9001 --name alpha
```

```bash
python scripts/mock_provider.py --port 9002 --name beta
```

```bash
python -m uvicorn app.main:app --port 8080
```

Then confirm it works:

```bash
curl -s localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer prism-sk-research-4d5e6f" \
  -H "Content-Type: application/json" \
  -d '{"model":"fast","messages":[{"role":"user","content":"What is a message queue?"}]}' -D-
```

The `x-prism-*` response headers tell you what happened:

```
x-prism-provider: alpha/alpha-small     which upstream actually answered
x-prism-cache: miss                     hit or miss
x-prism-fallback: false                 true if the primary provider was down
x-prism-cost-usd: 0.00001230            what this request cost
```

**Ops console:** <http://localhost:8080/console> — usage per team, spend against
budget, cache hit rate, provider health, and a live request log showing each
routing decision. The admin token is `prism-admin-token`.

No database to install: Prism uses SQLite by default and creates `prism.db` on
startup. Point `PRISM_DATABASE_URL` at Postgres to switch. See `.env.example`.

### Seeded demo tenants

Loaded from `data/seed_keys.json` on every startup, values preserved verbatim.

| Key | Team | Budget | Rate limit | Models | Cache |
|---|---|---|---|---|---|
| `prism-sk-search-1a2b3c` | search | $50 | 60/min | `fast` | on @ 0.92 |
| `prism-sk-research-4d5e6f` | research | $500 | 300/min | `fast`,`smart`,`auto` | off |
| `prism-sk-free-7g8h9i` | free-tier | $5 | 10/min | `fast` | on @ 0.85 |
| `prism-sk-budget-demo-0j1k2l` | budget-demo | $0.00001 | 60/min | `fast` | off |

The budget-demo key's cap is deliberately smaller than one request, so budget
rejection can be demonstrated in one command.

---

## The five-minute demo

Each block shows one capability. `-D-` prints the response headers.

<details open>
<summary><b>1. Streaming — tokens arrive as they are generated</b></summary>

```bash
curl -N -s localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer prism-sk-research-4d5e6f" \
  -H "Content-Type: application/json" \
  -d '{"model":"fast","stream":true,"messages":[{"role":"user","content":"Explain server-sent events."}]}'
```

Chunks arrive ~25ms apart and the stream ends with `data: [DONE]`. Nothing is
buffered — the gateway forwards each chunk the moment it arrives.
</details>

<details open>
<summary><b>2. Smart routing — same alias, two different models</b></summary>

```bash
# Trivial question -> cheap model
curl -s -D- -o /dev/null localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer prism-sk-research-4d5e6f" -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is the capital of France?"}]}' | grep x-prism-provider

# Short but genuinely hard -> strong model
curl -s -D- -o /dev/null localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer prism-sk-research-4d5e6f" -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Prove that the square root of 2 is irrational."}]}' | grep x-prism-provider
```

Returns `alpha/alpha-small` then `alpha/alpha-large`. The second prompt is
*shorter* than the first — length would have got this backwards.
</details>

<details open>
<summary><b>3. Semantic cache — a paraphrase is free</b></summary>

```bash
ask() { curl -s -D- -o /dev/null localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer prism-sk-search-1a2b3c" -H "Content-Type: application/json" \
  -d "{\"model\":\"fast\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}]}" \
  | grep -E "x-prism-(cache|cost)"; }

ask "How do I export my invoices as a CSV file?"      # miss, costs money
ask "What are the steps to export my invoices to CSV?" # HIT, costs 0
ask "Compare TCP and UDP for game servers."            # miss - different question
```

Different words, different word order, same question — served from cache. Then
prove it does not leak across tenants by asking the *same* question as another
team:

```bash
curl -s -D- -o /dev/null localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer prism-sk-free-7g8h9i" -H "Content-Type: application/json" \
  -d '{"model":"fast","messages":[{"role":"user","content":"How do I export my invoices as a CSV file?"}]}' | grep x-prism-cache
```

`miss` — as it must be.
</details>

<details open>
<summary><b>4. Budget and rate limits — two different refusals</b></summary>

```bash
# Budget: run twice. The second is refused.
curl -s localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer prism-sk-budget-demo-0j1k2l" -H "Content-Type: application/json" \
  -d '{"model":"fast","messages":[{"role":"user","content":"hello"}]}'

# Rate limit: 15 requests against a 10/min key.
for i in $(seq 1 15); do curl -s -o /dev/null -w "%{http_code} " localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer prism-sk-free-7g8h9i" -H "Content-Type: application/json" \
  -d '{"model":"fast","messages":[{"role":"user","content":"hi '$i'"}]}'; done; echo
```

Budget gives `402 budget_exceeded`; the rate limit gives `429
rate_limit_exceeded`. Different codes because they need different reactions —
waiting fixes one and not the other.
</details>

<details open>
<summary><b>5. Live failover — kill a provider mid-demo</b></summary>

```bash
curl -s -X POST localhost:9001/admin/config -d '{"mode":"down"}'   # alpha is now dead

curl -s -D- -o /dev/null localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer prism-sk-research-4d5e6f" -H "Content-Type: application/json" \
  -d '{"model":"fast","messages":[{"role":"user","content":"still working?"}]}' | grep x-prism

curl -s -X POST localhost:9001/admin/config -d '{"mode":"ok"}'     # bring it back
```

The request still succeeds, served by `beta/beta-small` with
`x-prism-fallback: true`. Callers see no error. Works for streaming too.
</details>

### Reproduce every claim at once

```bash
pkill -f "uvicorn app.main:app"; rm -f prism.db*      # verification needs a clean store
python -m uvicorn app.main:app --port 8080 &
python scripts/verify.py
```

This runs the provided smoke test, the routing eval, a concurrency burst, a live
failover, and the cache checks, then writes [VERIFICATION.md](VERIFICATION.md).
Every number in that report comes from a command the script actually ran.

> It refuses to run against a gateway that has already served traffic. That is
> deliberate: the smoke test salts its prompts with a random uuid to dodge the
> cache, which defeats an *exact-match* cache but not a semantic one — `(a) What
> is a load balancer?` and `(b) What is a load balancer?` are ~0.98 similar. A
> warm store would produce a report full of real-looking numbers that mean
> nothing.

---

## How a request flows

Every request walks these nine steps in this order. The whole sequence is
readable top-to-bottom in [`app/api/chat.py`](app/api/chat.py).

```
 1. parse       is the body well-formed?              400 invalid_request_error
 2. authenticate is this a real, active key?           401 authentication_error
 3. allowlist   may this key use this model?           403 model_not_allowed
 4. rate limit  quota left this minute?                429 rate_limit_exceeded
 5. budget      money left this month?                 402 budget_exceeded
 6. route       which model actually serves this?      404 not_found_error
 7. cache       answered this before?  ──── hit ──▶ return, $0, done
 8. dispatch    call upstream, retry, fail over        502 upstream_error
 9. meter       compute cost, update totals, log it
```

**Why this order.** Cheap local checks come first, so a request with a bogus key
never touches a database counter. The rate limit is checked *before* the cache,
not after: a cache hit is still a request the gateway served, and a tenant that
can hammer the cache without limit can still hammer the gateway.

Every exit path — success, cache hit, or any rejection — writes exactly one
request-log row. That is what makes the log trustworthy: if a caller says "my
request vanished", it is in there with a reason.

## Reading the code

Each module owns one decision, so `chat.py` is sequencing and almost no policy.

| File | Responsibility |
|---|---|
| [`app/api/chat.py`](app/api/chat.py) | **Start here.** The nine steps above |
| [`app/limits.py`](app/limits.py) | Auth, allowlist, rate limit, budget — the correctness-critical part |
| [`app/routing/classifier.py`](app/routing/classifier.py) | Prompt difficulty judgement for `auto` |
| [`app/cache/semantic_cache.py`](app/cache/semantic_cache.py) | Scoping, similarity search, TTL |
| [`app/cache/embedder.py`](app/cache/embedder.py) | Text → vector, with a dependency-free fallback |
| [`app/dispatch.py`](app/dispatch.py) | Alias resolution, retries, failover |
| [`app/providers/`](app/providers/) | The adapter boundary: no SDK leaks into core code |
| [`app/pricing.py`](app/pricing.py) | Cost computation |
| [`app/models.py`](app/models.py) | The six persisted entities |
| [`app/api/admin.py`](app/api/admin.py) | Usage, logs, cache stats, provider health |

Every file opens with a docstring explaining *why* it is built the way it is,
not just what it does. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers the
data model, the measured cache calibration, and the concurrency reasoning.

---

## API reference

### Data plane

#### `POST /v1/chat/completions`

OpenAI-compatible. Authenticate with `Authorization: Bearer <virtual-key>`.

```json
{
  "model": "fast",
  "messages": [{"role": "user", "content": "What is a message queue?"}],
  "stream": false
}
```

`model` accepts an alias (`fast`, `smart`, `auto`) or a concrete model name
(`alpha-small`). Unknown parameters such as `temperature` are forwarded upstream
untouched — a gateway that only passes through fields it was taught about would
silently drop provider features.

**Response headers, on every response:**

| Header | Meaning |
|---|---|
| `x-prism-provider` | who served it, `provider/model` |
| `x-prism-cache` | `hit` or `miss` |
| `x-prism-fallback` | `true` only when a fallback provider served it |
| `x-prism-cost-usd` | computed cost; `0` on cache hits |

With `"stream": true` the response is `text/event-stream`, forwarded chunk by
chunk and terminated by `data: [DONE]`. **`x-prism-cost-usd` is absent on
streaming responses** — headers are sent before the body, and the provider only
reports token counts in its final chunk. The cost is still recorded in the
request log and the usage API; it just arrives late.

#### Errors

Every rejection returns an OpenAI-shaped body with a distinct `type`, so a
client can tell what to do from the body alone:

```json
{"error": {"message": "...", "type": "budget_exceeded", "code": "budget_exceeded"}}
```

| Status | `type` | Meaning |
|---|---|---|
| 400 | `invalid_request_error` | malformed body, empty `messages` |
| 401 | `authentication_error` | missing, unknown, or disabled key |
| 403 | `model_not_allowed` | valid key, model not on its allowlist |
| 404 | `not_found_error` | no such model or alias on this gateway |
| 429 | `rate_limit_exceeded` | per-minute quota spent — retry shortly |
| 402 | `budget_exceeded` | monthly budget spent — retrying will not help |
| 502 | `upstream_error` | every provider failed, after retries and failover |
| 503 | `service_overloaded` | the gateway itself is out of capacity |

402 vs 429 is a deliberate split. Both mean "not right now", but only one is
fixed by waiting, and a client's retry logic needs to tell them apart.

### Admin plane

Authenticate with `X-Admin-Token: prism-admin-token`.

| Endpoint | Returns |
|---|---|
| `GET /admin/usage?key=&from=&to=` | spend and token totals; reconciles with the cost headers |
| `GET /admin/logs?key=&status=&limit=` | recent requests, newest first, including rejections |
| `GET /admin/cache/stats?key=` | hits, misses, hit rate, stored entries |
| `GET /admin/keys` | tenants, their policy, and spend against budget |
| `GET /admin/providers/health` | per-provider health from a probe plus recent traffic |
| `GET /admin/config` | active aliases, chains, retry policy (never credentials) |
| `GET /health` | liveness, unauthenticated |

The two planes use different credentials on purpose: a virtual key can only ever
affect its own tenant, while the admin token sees across all of them. Giving
both audiences one credential is how cross-tenant leaks happen.

---

## Design decisions

### 1. Rate limiting without over-admission

The obvious rate limiter reads a counter, compares it to the limit, and writes
it back. Under concurrency this is broken: ten simultaneous requests can all read
"9 used, limit 10" before any of them writes, and all ten are admitted.

Prism never reads-then-writes. The check and the increment are a **single SQL
statement** whose `WHERE` clause carries the limit, so the database decides:

```sql
INSERT INTO rate_limit_windows (virtual_key, window_start, count)
VALUES (:key, :window, 1)
ON CONFLICT (virtual_key, window_start) DO UPDATE
    SET count = rate_limit_windows.count + 1
    WHERE rate_limit_windows.count < :limit
RETURNING count
```

No row returned means the limit was already reached. Two requests cannot both
claim the last slot. Usage totals are incremented the same way, so concurrent
cost recording cannot lose an update.

Verified two ways: a unit test fires 50 concurrent attempts at a limit of 10 and
asserts exactly 10 succeed, and the provided load test reports **10 accepted
against a limit of 10** with no over-admission.

### 2. Budgets, and the overshoot we accept

A request's true cost is unknowable until the provider reports its token counts,
which happens *after* the work is done. So the budget is pre-checked against
spend so far and reconciled afterwards.

The honest consequence: requests admitted while spend is just under the cap can
push it over, bounded by the cost of whatever was in flight at that moment. The
window is wider for streaming, which can run for seconds. The alternative —
holding every request until all in-flight ones settle — would serialise the
gateway. This is documented rather than hidden, and the same trade-off is made
by every gateway that doesn't serialise.

### 3. Smart routing: length is a trap

The intuitive proxy for difficulty is prompt length, and it is wrong. The eval
set punishes it from both directions:

- *"Prove that the square root of 2 is irrational."* — 9 words, needs a proof.
- A 14-line on-call roster ending *"Who is on call this Thursday?"* — 90 words,
  needs one table lookup.

Length ranks the second as harder. That is exactly backwards.

**What Prism does instead: it separates the task from the material.** A prompt is
an instruction plus, often, a pile of data to apply it to. The data can be
arbitrarily large without making the task harder. So for long prompts the
classifier keeps the opening and closing sentences — where instructions actually
live — and scores *those* for the language of hard work: `prove`, `design …
explain your trade-offs`, `in order of likelihood`, `estimate how many`.

It defaults to the cheap tier and escalates only on evidence, because a wrong
escalation costs 20× while a wrong de-escalation costs one retry.

| | Prism | Length-only baseline |
|---|---|---|
| Overall | **20/20** | 11/20 |
| Trap cases only | **8/8** | 1/8 |

```bash
python scripts/routing_eval.py --verbose
```

The signal weights were hand-built from the *language* of hard requests, not
fitted to the eval file — and the test suite includes 13 prompts written
separately from it, to check the classifier generalises rather than memorises.
Every `auto` decision writes its reasoning to the request log, so an operator
asking "why did this cost me 20×?" gets an answer.

### 4. Cache scoping and tenant isolation

Three things must match before two prompts are even compared:

1. **`virtual_key`** — the cache is per tenant. Tenant A's answer is never a
   candidate for tenant B's question, even if the prompts are byte-identical.
   This predicate is in every lookup query; it is what makes a shared cache safe.
2. **model** — a `fast` answer is not served to a `smart` request.
3. **conversation prefix** — only the *last user message* is embedded; earlier
   turns are hashed into the scope. So a follow-up like *"What happens when it's
   exhausted?"* can only match another request that had the same conversation
   before it, because on its own that question means different things in
   different conversations.

**The threshold is real per-tenant policy, not decoration.** Measured with the
default model, the paraphrase pair *"If I bought an annual plan, can I get my
money back?"* / *"What is the refund policy for annual plans?"* scores **0.869**:
a hit for the free tier (0.85) and a miss for the search team (0.92). Both are
correct — a team that wants fewer false matches accepts fewer hits. The full
measured similarity table is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Cache hits cost **$0**. Against the local mocks they are only marginally faster
(10ms vs 11ms) because the mocks answer instantly; against a real provider taking
seconds, the latency saving is the larger part. Here, the saving is cost.

### 5. Retries, failover, and the streaming rule

Two different recovery mechanisms for two different problems:

- **Retry** — for transient trouble (timeout, 429, one-off 500). The same model
  is asked again after exponential backoff (200ms, 400ms, 800ms). Retrying a
  struggling provider in a tight loop turns its bad minute into its bad afternoon.
- **Failover** — for a provider that is genuinely broken. Once a model exhausts
  its retries, the next model in the alias chain is tried: a *different provider*.
  `fast` is `alpha-small` with `beta-small` behind it.

Non-retryable failures (a 4xx that isn't a 429) skip straight to the next model —
asking a provider twice for a model it doesn't serve is just latency.

**The streaming rule:** failover works for streams because the stream is opened
*before* the response headers are committed. But the moment the first byte
reaches the client, the gateway stops failing over — splicing two different
answers together would be worse than a truncated stream the client can detect.

Adding a real provider is a config edit, not a code change: one adapter class
already covers every OpenAI-compatible provider. Core code never imports a
provider SDK.

---

## Testing

```bash
pytest                      # 138 tests, ~5s
pytest tests/test_limits.py # the concurrency-critical ones
```

| File | Covers |
|---|---|
| `test_limits.py` | auth, allowlist, rate limiting, budgets — with real concurrent tasks |
| `test_cache.py` | scoping, tenant isolation, thresholds, TTL, embedders |
| `test_routing.py` | traps, held-out prompts, the baseline comparison |
| `test_dispatch.py` | retries, failover, the mid-stream safety rule |
| `test_pricing.py` | cost computation and header formatting |
| `test_api.py` | end-to-end through the real ASGI app |

The concurrency tests run genuinely concurrent tasks against a real database
rather than asserting on mocks — the bug being guarded against is a race, and a
mock cannot have one.

---

## Known limitations

Stated plainly, because a gateway whose limits you don't know is one you can't
operate.

- **Budgets can overshoot slightly** by the cost of in-flight requests, as
  described above. Wider for streams.
- **Rate limiting is a fixed window**, so up to 2× the limit can pass across a
  window boundary. Within any single window the limit is exact, which is what the
  over-admission check measures. A sliding window would fix this.
- **Cache similarity is a linear scan** over the scoped candidate set. That set is
  one tenant + one model + one conversation prefix, so it stays small, but it is
  not an index. pgvector is the scaling path and changes one method.
- **Cache correctness is the caller's judgement.** Prism will happily serve a
  cached answer to *"What is the current status of the payments service?"*.
  Time-sensitive prompts should not be sent to a cache-enabled key; per-request
  opt-out is not implemented.
- **Single-instance assumptions are untested.** Rate-limit and cache state live in
  the database, so a second gateway process would share them correctly, but this
  has not been load-tested multi-instance.
- **The admin plane is one shared token.** Fine for a single-operator demo; a real
  deployment needs per-operator identities and an audit trail.
- **Prompt bodies are not logged** by default (`PRISM_LOG_PROMPT_BODIES`). The
  request log holds metadata only — enough to debug traffic, without turning the
  log into a copy of every customer's data.
- **Postgres is supported but SQLite is what's exercised.** The correctness-critical
  SQL is written to behave identically on both, and the test suite runs on SQLite.
