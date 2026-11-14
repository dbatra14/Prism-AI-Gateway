# Demo Video Walkthrough

A screen-recorded tour of Prism, kept under five minutes. This document is the
shot list and narration used for the recording, so anyone can reproduce the
same demo.

> **Video link:** _add the recording URL here after uploading (e.g. YouTube
> unlisted / Google Drive)._

## Pre-flight (before recording)

1. Start the stack:

   ```bash
   docker compose up --build
   ```

   (or, without Docker: `bash scripts/run_local.sh`)

2. Wait for `Prism ready`, then open:

   ```text
   http://localhost:8080/console/chat.html
   ```

3. Keep the Terminal window **behind** the browser — it must never be visible,
   because a local `.env` (containing provider secrets) is never shown on
   screen.

## Shot list (~4 minutes)

| # | Shot | Time | Say |
|---|------|------|-----|
| 1 | Browser on the chat console | 0:15 | "This is Prism, an AI gateway. Apps don't call AI providers directly — they call Prism, which authenticates the key, routes to a model, checks the cache, meters usage, and logs everything." |
| 2 | Chat: send "Explain what an API gateway does in two simple sentences." as **Search team / Fast** | 0:45 | "A real answer served through the gateway. The labels show the provider that answered, cache status, and the tracked cost." |
| 3 | Send the **exact same** prompt again | 0:30 | "Same question, second time: the cache label flips to **hit** and the cost is **0.00** — no provider call, no tokens billed." |
| 4 | Ops console (`/console/`) | 0:45 | "Every request is metered: totals, spend per tenant against budget, cache hit rate, provider health, and the live request log with routing decisions." |
| 5 | `docs/ARCHITECTURE.md` first diagram | 0:45 | "Request flow: authenticate → route (with retry/failover) → cache → meter → persist. Providers are adapters, so swapping real providers never changes the client API." |
| 6 | Docker: `docker compose ps` in a clean terminal | 0:30 | "The whole stack is containerized: gateway plus two OpenAI-compatible mock providers, healthchecks, and persistent volumes — `docker compose up` is the only command." |
| 7 | Close | 0:10 | "Prism: a practical, production-shaped AI gateway. Thanks for watching." |

## Recording checklist

- [ ] macOS: `Cmd+Shift+5` → select browser region → **Microphone: on**
- [ ] Voice level checked with a 5-second test clip
- [ ] No `.env`, keys, or Terminal visible at any point
- [ ] Final length under 5:00
