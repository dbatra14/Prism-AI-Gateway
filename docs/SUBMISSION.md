# Prism Submission — Simple Steps

You only need to do these three things.

For the exact recording steps and words to say, open [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

## 1. Start Prism

Open Terminal and run:

```bash
cd "$(git rev-parse --show-toplevel)"
bash scripts/run_local.sh
```

Keep Terminal open.

Open this page in your browser:

```text
http://localhost:8080/console/chat.html
```

Ask one question and click **Send message**. You should see a real answer.

## 2. Record a short video

Record your screen while you:

1. Ask Prism one question in the chat page.
2. Show the answer.
3. Click **Ops console** at the top right.
4. Show that the request appears in **Recent Requests**.

That is enough for a simple demo.

## 3. Upload to GitHub

Do not upload your `.env` file or API key.

When you are ready, ask Codex/OpenCode to help you commit and push the project. Before pushing, verify that `.env` is not listed by this command:

```bash
git status
```

## Important

Never show your API key in the recording, on GitHub, or in chat.
