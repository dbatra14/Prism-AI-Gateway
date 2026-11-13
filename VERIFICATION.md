# Prism — Verification Report

Generated `2026-07-31 19:48:01 UTC` by `python scripts/verify.py` against `http://localhost:8080`,
with both mock providers running. Every number below is captured from a command
this script ran; nothing is transcribed by hand.

**Overall: PASS**

| Check | Result | Detail |
|---|---|---|
| Smoke test (provided) | PASS | 18 passed, 0 warnings, 0 failed |
| Routing eval vs baseline | PASS | classifier 20/20 vs length-only 11/20 |
| Over-admission under burst | PASS | 10 accepted, limit 10 |
| Accounting reconciliation | PASS | client sums == usage API |
| Provider failover | PASS | streaming and non-streaming |
| Rejection taxonomy | PASS | 6 distinct error types |
| Semantic cache + isolation | PASS | paraphrase hit, no cross-tenant leak |

---

## 1. Smoke test

The provided `scripts/smoke_test.py`, unmodified.

```
[1] Non-streaming completion + header contract
  PASS  returns 200  (got 200)
  PASS  body has choices and usage
  PASS  header x-prism-provider present  (alpha/alpha-small)
  PASS  header x-prism-cache present  (miss)
  PASS  header x-prism-cost-usd present  (0.00001350)
  PASS  first request is a cache miss  (miss)

[2] Streaming
  PASS  returns 200  (got 200)
  PASS  content-type is text/event-stream  (text/event-stream; charset=utf-8)
  PASS  received multiple SSE chunks  (22 data lines)
  PASS  stream ends with [DONE]

[3] Auth
  PASS  invalid key rejected with 401/403  (got 401)

[4] Unknown model
  PASS  unknown model rejected with 4xx  (got 404)
  PASS  error body explains the problem  ({'error': {'message': "Unknown model or alias 'no-such-model-xyz'. Known: auto, )

[5] Semantic cache: exact repeat
  PASS  identical repeat is a cache hit  (hit)

[6] Semantic cache: paraphrases (each pair soft; at least one pair must hit)
  PASS  pair A paraphrase is a cache hit  (hit)
  PASS  pair B paraphrase is a cache hit  (hit)
  PASS  matching is semantic, not exact-only (at least one pair hit)

[7] Semantic cache: unrelated prompt
  PASS  unrelated prompt is a cache miss  (miss)

==================================================
  18 passed, 0 warnings, 0 failed
```

## 2. Smart routing accuracy

- **Prism classifier: 20/20**
- Length-only baseline: 11/20
- On trap cases only (short-but-hard, long-but-trivial): **8/8** vs baseline 1/8

The trap row is the meaningful one. Aggregate accuracy can look respectable
while a router fails every case designed to punish word counting.

```
Routing eval — 20 labelled cases from routing_eval.jsonl

==============================================================
  Prism classifier   20/20  (100%)
  Length-only        11/20  (55%)   <- baseline to beat

  On trap cases only (8 short-but-hard / long-but-trivial):
    Prism classifier 8/8
    Length-only      1/8
==============================================================

PASS: beats the length-only baseline by 9 cases.
```

## 3. Over-admission under concurrency

A burst of 30 requests at concurrency 10 against the free-tier key, whose limit is 10/min.

- Accepted: **10** (limit 10)
- Rate limited: 20
- **PASS** — no over-admission.

This is the check that catches a read-then-write rate limiter. The counter is
incremented by a single conditional `UPDATE ... WHERE count < limit`, so the
database decides admission and two requests cannot both claim the last slot.
See `app/limits.py`.

```
Bursting 30 requests at concurrency 10 ...

Burst finished in 0.2s
  accepted (200):      10
  rate limited (429):  20
  latency avg/p95 ms:  69 / 105

Client-side accounting totals - compare with your usage API for this key:
  accepted requests:   10
  prompt tokens:       120
  completion tokens:   262
  sum of cost headers: 0.000071 USD (10 of 10 had a numeric header)

Over-admission check OK: 10 accepted <= limit 10.
```

## 4. Accounting accuracy under concurrency

40 requests at concurrency 20. The load test sums the `x-prism-cost-usd` headers
it received; the gateway's own usage API is queried before and after. They must agree.

| Metric | Client-observed | Gateway usage API | Match |
|---|---:|---:|:--:|
| Requests | 40 | 40 | yes |
| Prompt tokens | 480 | 480 | yes |
| Completion tokens | 1031 | 1031 | yes |
| Cost (USD) | 0.00069100 | 0.00069060 | yes |

Latency for that burst: **avg 72ms, p95 191ms**
(client-observed, against the local mock providers).

```
Bursting 40 requests at concurrency 20 ...

Burst finished in 0.2s
  accepted (200):      40
  rate limited (429):  0
  latency avg/p95 ms:  72 / 191

Client-side accounting totals - compare with your usage API for this key:
  accepted requests:   40
  prompt tokens:       480
  completion tokens:   1031
  sum of cost headers: 0.000691 USD (40 of 40 had a numeric header)
```

## 5. Live provider failover

`alpha` is taken down for real via the mock's failure-injection endpoint
(`POST /admin/config {"mode": "down"}`), not simulated in code.

| Stage | Served by | `x-prism-fallback` |
|---|---|---|
| alpha healthy | `alpha/alpha-small` | false |
| alpha down | `beta/beta-small` | **true** |
| alpha down, streaming | `beta/beta-small` | true |
| alpha restored | `alpha/alpha-small` | false |

With **both** providers down the gateway returns `502` `upstream_error` rather than hanging or returning a broken 200.

Failover works for streaming because the stream is opened before the response
headers are committed. Once the first byte reaches the client the gateway stops
failing over deliberately — splicing two different answers together would be
worse than a truncated stream the client can detect (`app/dispatch.py`).

## 6. Rejection taxonomy

Every rejection carries a distinct `type`, so a caller can tell what to do about
it from the body alone. A 429 for rate limiting and a 402 for an exhausted budget
need opposite responses: one says retry shortly, the other says retrying is futile.

| Case | Status | `error.type` |
|---|---:|---|
| invalid key | 401 | `authentication_error` |
| model not on allowlist | 403 | `model_not_allowed` |
| unknown model | 404 | `not_found_error` |
| empty messages | 400 | `invalid_request_error` |
| budget exhausted | 402 | `budget_exceeded` |
| rate limit exceeded | 429 | `rate_limit_exceeded` |

## 7. Semantic cache and tenant isolation

| Request | `x-prism-cache` | Expected |
|---|---|---|
| Original question | `miss` | miss |
| Paraphrase, sharing almost no words | **`hit`** | hit |
| Unrelated question | `miss` | miss |
| Original question, **different tenant** | **`miss`** | miss |

The paraphrase pair is *What are the steps to export my invoices to CSV?* against
*How do I export my invoices as a CSV file?* — the same question with different
wording and word order. An exact-match cache cannot hit this; matching on the
meaning of the prompt is the entire point.

The last row is the security-relevant one: a byte-identical prompt from another
tenant is a miss, because `virtual_key` is a predicate on every cache query.

Note that the threshold is genuine per-tenant policy, not decoration. The harder
pair *If I bought an annual plan, can I get my money back?* / *What is the refund
policy for annual plans?* measures **0.869** with the default model: it is a hit
for the free tier (threshold 0.85) and a miss for the search team (0.92). Both are
correct — a team that wants fewer false matches pays for more of them.

- Cache hit cost: `0.00000000` USD (hits are free)
- Median latency: **10ms on a hit vs 11ms on a miss**
- Cache hit rate across this run: 0.2206  (9 entries stored)

## 8. Platform totals after this run

| Key | Team | Requests | Tokens (in/out) | Spend | Budget |
|---|---|---:|---:|---:|---:|
| `prism-sk-search-1a2b3c` | search | 15 | 78/263 | $0.000042 | $50.0 |
| `prism-sk-research-4d5e6f` | research | 44 | 493/1100 | $0.000726 | $500.0 |
| `prism-sk-free-7g8h9i` | free-tier | 19 | 197/477 | $0.000160 | $5.0 |
| `prism-sk-budget-demo-0j1k2l` | budget-demo | 1 | 4/17 | $0.000011 | $1e-05 |

Total across all tenants: **79 requests, $0.000939**, 21 served from cache.

## Known limitations

- **Budget can overshoot slightly.** A request's true cost is only known after the
  provider reports token counts, so the budget is pre-checked against spend so far
  and reconciled afterwards. Requests admitted while spend is just under the cap can
  push it over, bounded by the cost of the requests in flight. Wider for streams,
  which run for seconds. The alternative — holding every request until all in-flight
  ones settle — would serialise the gateway.
- **Rate limiting is a fixed window**, so up to 2x the limit can pass across a window
  boundary. Within any single window the limit is exact, which is what the
  over-admission check measures.
- **Cache similarity is computed in-process** over the scoped candidate set. That set
  is one tenant + one model + one conversation prefix, so it stays small, but it is a
  linear scan. pgvector is the scaling path and changes one method.
- **Rate-limit and cache state are single-instance.** Both live in the database, so a
  second gateway process shares them correctly, but neither has been load-tested
  multi-instance.
- **The admin plane is one shared token.** Fine for a single-operator demo; a real
  deployment wants per-operator identities and an audit trail.
