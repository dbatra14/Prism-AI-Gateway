# Prism — Architecture

Companion to the [README](../README.md). This covers the data model, the
concurrency reasoning, the measured cache calibration, and the trade-offs behind
each choice.

---

## 1. Shape of the system

```
                      ┌──────────────────────────────────────────┐
  caller ──Bearer──▶  │  app/api/chat.py    the nine steps       │
                      │      │                                   │
                      │      ├─▶ app/limits.py        2·3·4·5    │
                      │      ├─▶ app/routing/         6          │
                      │      ├─▶ app/cache/           7          │
                      │      ├─▶ app/dispatch.py      8 ─────────┼──▶ providers
                      │      └─▶ app/pricing.py       9          │
                      └──────────────────┬───────────────────────┘
                                         ▼
                              one store: keys · usage · logs · cache
```

`chat.py` sequences; every other module owns exactly one decision. That is why
the request lifecycle reads top-to-bottom in a single function while the file
holds almost no policy of its own.

### Why one store

Keys, usage counters, request logs and cache entries all live in one database.
Adding Redis for counters would be conventional, but it would split the
correctness-critical state across two systems that cannot participate in the same
atomic operation. Admission control depends on a *single* statement doing both
the check and the increment (§3), and that is only possible if the counter and
the decision live together.

SQLite is the default so the gateway runs with no setup and the demo is
reproducible on any machine. `PRISM_DATABASE_URL` switches to Postgres; the
correctness-critical SQL is written to behave identically on both.

### The provider boundary

Core code never imports a provider SDK. It only ever calls `Provider.chat()` or
`Provider.open_stream()` ([`app/providers/base.py`](../app/providers/base.py)).

The adapter's second job is **translating every upstream failure into the
gateway's own taxonomy**, each tagged retryable or not:

| Adapter raises | When | Retryable |
|---|---|---|
| `ProviderTimeout` | no answer inside the timeout | yes |
| `ProviderUnavailable` | connection refused/reset, or 5xx | yes |
| `ProviderRateLimited` | upstream 429 | yes |
| `ProviderBadRequest` | other 4xx — malformed, unknown model | no |

The dispatcher retries purely on that flag, so it never needs to know whether a
given provider signals overload with a 429, a 503, or a dropped connection.
Adding a real provider means writing one adapter class — nothing in routing,
metering, caching or limits changes.

**Every upstream call is bounded by an explicit timeout.** A provider that
accepts a connection and then goes silent is the worst failure mode there is:
without a timeout it holds a gateway worker forever.

---

## 2. Data model

Six entities ([`app/models.py`](../app/models.py)).

| Entity | Grain | Purpose |
|---|---|---|
| `VirtualKey` | one per tenant | budget, rate limit, allowlist, cache policy |
| `ModelPrice` | one per model | USD per 1M input/output tokens |
| `UsageRecord` | key × month | running spend and token totals |
| `RateLimitWindow` | key × minute | the request counter |
| `RequestLog` | one per request | the full trace, including rejections |
| `CacheEntry` | per key | prompt embedding + the response it produced |

**`UsageRecord` and `RateLimitWindow` are aggregates, not views over the log.**
Budget and rate-limit checks run on every request, so they must be a single
indexed row read — never a scan over the log table. This is the difference
between a gateway that stays fast as the log grows and one that doesn't.

`RequestLog` records rejections too. A request that never reached a provider is
exactly what an operator needs to see when a tenant reports errors.

### Timestamps

All timestamps are **naive UTC**. SQLite's `DATETIME` discards `tzinfo` on write
and returns naive values on read, so a `timezone=True` column would quietly mean
different things on SQLite and Postgres. Storing naive UTC everywhere makes the
two backends behave identically; the API appends the `Z` when serialising.

*(Found the hard way: the first version used timezone-aware columns and would
have silently written offsets on one backend and dropped them on the other.)*

### Privacy

Prompt and response bodies are **not** written to the request log by default.
The log holds metadata — key, models, tokens, cost, cache result, latency,
routing reason — which is enough to debug traffic without turning the log into a
copy of every customer's data. `PRISM_LOG_PROMPT_BODIES` exists but the retention
and privacy implications are the operator's to accept.

---

## 3. Concurrency: the correctness core

### The race

Read-then-write admission control is broken under concurrency:

```
request A: read count=9  ─┐
request B: read count=9  ─┤ both see "under the limit of 10"
request A: write count=10 │
request B: write count=10 ┘ two admitted into one slot; counter says 10
```

This passes every sequential test and fails in production.

### The fix

Never read-then-write. Put the limit in the `WHERE` clause and let the database
decide, in one statement:

```sql
INSERT INTO rate_limit_windows (virtual_key, window_start, count)
VALUES (:key, :window, 1)
ON CONFLICT (virtual_key, window_start) DO UPDATE
    SET count = rate_limit_windows.count + 1
    WHERE rate_limit_windows.count < :limit
RETURNING count
```

- First request of a minute: the `INSERT` succeeds with `count = 1`.
- Later requests: the insert conflicts, and `DO UPDATE` fires *only while* the
  stored count is below the limit.
- At the limit: the `WHERE` fails, no row is updated, `RETURNING` is empty →
  reject.

Usage totals use the same insert-or-increment shape without the condition, so
concurrent cost recording cannot lose an update.

The commit happens immediately rather than at end of request: concurrent requests
must see the increment, or they would all still read a stale count.

### SQLite specifics

WAL mode plus `busy_timeout=10000`. Without them a concurrent burst fails on
`database is locked` rather than on anything meaningful. SQLite serialises
writers, which is exactly the guarantee the statement above needs.

### Connection pool sizing

The default pool (5 + 10 overflow) is too small here and fails confusingly:
under a burst, requests block waiting for a connection and surface as slow 500s
that *look* like provider timeouts. Two things drive the requirement up:

- a streaming request holds its request-scoped session for the whole stream, and
  its finaliser opens a second one to record usage;
- admission control commits several times per request.

Pool is 20 + 40 with a **5-second** `pool_timeout` — if the pool really is
exhausted, failing fast is far more useful than a 30-second hang, and it surfaces
as `503 service_overloaded` rather than a generic error.

*(This was a real bug: at concurrency 20, 27 of 40 requests failed after exactly
30 seconds. See the README's testing section.)*

### Verified

- Unit test: 50 concurrent attempts at a limit of 10 → exactly 10 admitted.
- Unit test: 60 concurrent cost recordings → totals sum exactly, none lost.
- Load test: 10 accepted against a limit of 10; no over-admission.
- Load test: 100 requests at concurrency 50, zero unhandled errors, totals
  reconcile with the usage API.

---

## 4. Cost accounting

```
cost = prompt_tokens/1M × input_price + completion_tokens/1M × output_price
```

Always using the token counts the **provider** reported, never a local estimate.
If the gateway guessed, its numbers would drift from the provider's invoice and
budgets would be enforced against a number that isn't real money.

An unpriced model costs 0 rather than raising — a missing price row must not turn
a successful completion into a client-visible failure. The gap shows up in the
usage API instead, where an operator will see it.

`x-prism-cost-usd` is formatted to fixed 8-decimal precision, never scientific
notation: mock prices are small enough that `repr()` would emit `1e-05`.

**Cache hits record tokens but zero cost.** The tokens were served to the caller,
so counting them keeps the usage API reconcilable with response bodies; the cost
is zero because no money was spent. That gap between tokens served and dollars
spent *is* the cache's value, and the console shows it.

### Reconciliation

The load test sums the `x-prism-cost-usd` headers it received; `/admin/usage` is
queried before and after. They agree to the precision the load test prints (6
decimal places — the residual is its own display rounding, not drift).

---

## 5. Semantic cache

### Why embeddings

Callers rarely repeat themselves exactly. Hashing the prompt catches only literal
repeats; the cache exists to catch *the same question asked differently*.

An embedding maps text to a vector whose direction encodes meaning, so two
phrasings of one question land close together. Vectors are L2-normalised on the
way out, which makes cosine similarity a plain dot product — cheap, and it means
the threshold has the same meaning regardless of which backend produced the
vector.

### Choosing the model and threshold — measured, not guessed

Cosine similarity for pairs that *should* hit versus pairs that *must not*,
across three candidate models. The critical near-miss is "reset my **password**"
against "reset my **two-factor authentication**" — similar words, different
question, and serving one for the other would be a real bug.

| Pair | bge-small (default) | bge-base | MiniLM-L6 |
|---|---|---|---|
| *reset password* / *steps to reset dashboard password* | **0.983** | 0.965 | 0.951 |
| *refund policy for annual plans* / *can I get my money back* | **0.869** | 0.866 | 0.701 |
| password vs **two-factor** reset (must miss) | 0.849 | 0.833 | 0.757 |
| password reset vs TCP/UDP (must miss) | 0.419 | 0.348 | −0.135 |
| message queue vs load balancer (must miss) | 0.598 | 0.531 | 0.233 |
| **Separation margin** | **+0.021** | **+0.033** | **−0.056** |

Conclusions:

- **MiniLM fails outright** — negative margin, meaning no single threshold can
  separate paraphrases from near-misses. A popular default that would have been
  the wrong choice here.
- **bge-base separates best** but is ~3× the download for +0.012 of margin.
- **bge-small is the default**: adequate separation at ~130MB. Switching is one
  config line (`PRISM_EMBEDDING_MODEL`).

The seeded per-tenant thresholds are kept verbatim, and they land sensibly:

| Threshold | Result with bge-small |
|---|---|
| 0.92 (search) | high-overlap paraphrase hits; low-overlap pair misses; near-miss safely excluded |
| 0.85 (free-tier) | both paraphrase pairs hit; near-miss (0.849) excluded by 0.001 |

The free tier's margin over the near-miss is genuinely thin. That is the correct
shape of the trade-off — a looser threshold buys more hits and accepts more risk
of a wrong one — but it is worth knowing rather than discovering.

### The lexical fallback is honest about being worse

`PRISM_EMBEDDING_BACKEND=hash` runs with no download using hashed word and
character-trigram features. Measured on the same pairs it scores 0.588 and
**0.209** on the two paraphrases while scoring **0.784** on the near-miss — the
ordering is inverted, so no threshold works. It exists so the gateway still boots
without network access, not as an equivalent option.

### Scoping

Three predicates before two prompts are compared: `virtual_key`, `model`, and
`prefix_hash`.

Only the **last user message** is embedded. Earlier turns are hashed into
`prefix_hash`, which is part of the scope — so a follow-up can only match another
request that had the same conversation before it. *"What happens when it's
exhausted?"* means different things after "connection pooling" and after "thread
pools", and matching those to each other would serve a confidently wrong answer.

Requests with no usable user message bypass the cache entirely rather than
caching against a blank key.

### Storage and eviction

Embeddings are stored as raw float32 bytes. Lookup loads the scoped candidate
set and computes similarity in-process with numpy. Because the scope is one
tenant + one model + one conversation prefix, that set is small — a bounded
linear scan, not a search over every entry.

pgvector is the scaling path when a tenant's working set outgrows that, and it
changes exactly one method (`SemanticCache.lookup`). It is deliberately *not*
implemented: it would add a dependency and a Postgres requirement to buy
performance this workload doesn't need yet.

Eviction is TTL-based (24h default), swept by a background task rather than
inline — cleanup should not be paid for by one unlucky request. A dimension
mismatch (from switching models) degrades to a miss rather than crashing.

---

## 6. Routing

Full reasoning in the README (§3) and
[`app/routing/classifier.py`](../app/routing/classifier.py). The architectural
points:

- **The classifier is pure text analysis** — no model call, no embedding, no
  network. Routing adds no measurable latency to the hot path. An LLM-judge
  router would be more capable and would put a model call in front of every
  request; that cost is not obviously worth it when a feature classifier already
  scores 20/20.
- **Instruction extraction** is the mechanism that defeats both trap families:
  for long prompts, keep the first and last sentences (where instructions live)
  and drop the bulk material. That is why a 90-word roster with a lookup question
  routes cheap and a 9-word proof routes expensive.
- **Default cheap, escalate on evidence** — asymmetric costs (20× up, one retry
  down) justify an asymmetric default.
- **`auto` re-checks the allowlist** after picking a tier. A key allowed `auto`
  but not `smart` must not reach a smart model through the back door. Rather than
  reject — the caller asked for "whatever is appropriate", not a specific model —
  it is served on the best tier the key *is* allowed, and the log says so.
- **Every decision is logged with its reasoning**, so "why did this cost 20×?"
  has an answer without re-running anything.

The eval labels are used only for grading, never inside the routing logic, and
the test suite includes 13 prompts written independently of the eval file to
check generalisation rather than memorisation.

---

## 7. Streaming

`"stream": true` relays chunks provider → gateway → client with nothing held
back. Three details make that true:

1. **The stream is opened before the response is constructed**, because
   `x-prism-provider` must be on the headers and headers precede the body. This
   ordering is also what allows failover on a stream.
2. **Chunks are forwarded the instant they arrive.** The adapter *reads* each
   chunk on the way past to pick up the `usage` object the provider sends in its
   final chunk, but nothing accumulates.
3. **`X-Accel-Buffering: no`** tells reverse proxies not to buffer, which would
   defeat the whole thing.

Cost is recorded in a finaliser after the stream closes, on its own database
session — the request-scoped one is long gone by then.

**A cache hit on a streaming request is replayed as SSE**, so the caller's
parsing path is identical whether or not the cache answered.

Verified: chunks arrive ~25ms apart matching the mock's pacing, spread across
403ms rather than delivered as one burst.

### The mid-stream rule

Once the first byte reaches the client, the gateway stops failing over. Switching
providers mid-response would splice two different answers together — worse than a
truncated stream the client can detect. `StreamHandle.chunks_sent` tracks this,
and the dispatcher re-raises instead of retrying once it is non-zero.

---

## 8. What I would do next

In order of value:

1. **Sliding-window rate limiting** — removes the 2×-at-the-boundary artifact.
2. **Per-request cache opt-out** — the gateway cannot know that "what is the
   current status of the payments service?" must never be cached. The caller can.
3. **Provider circuit breaker** — health data is already collected and surfaced;
   it isn't yet used to *skip* a known-bad provider before trying it.
4. **pgvector** — when a tenant's cache outgrows a linear scan.
5. **Per-operator admin identities** — replacing the shared token.
6. **Response-quality escalation** — the mock providers support a `[refuse]`
   trigger where a small model refuses and a large one answers, which is the hook
   for detecting a weak answer and retrying a tier up. Not implemented.
