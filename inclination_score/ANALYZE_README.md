# Analyze: score any uploaded chat

Branch: `ayesha-chat-feature`

## What it does

On the web app's **Analyze** page, a user can upload or paste any chat. It is scored by the
same trained model as the datasets. The page shows:

- the chance the customer buys, after every message (the score curve)
- turning points: where interest rose or dropped sharply, and what triggered it
- a status: Hot, Rising, Cooling or Stalled

Before this change, the page was a **preview** that used simple keyword rules in the browser.
It now uses the real model.

## How to use it

```
cd inclination_score
make all        # first time only: prepares data and trains the models
make api        # terminal 1: starts the data server

cd web
npm run dev     # terminal 2: open the printed URL and go to Analyze
```

1. Upload a `.txt`, `.csv` or `.json` file, or paste lines like `Rep: Hello`.
2. Mark each speaker as **Customer** or **Sales side**.
3. Pick the kind of conversation: sales, price negotiation, or persuasion.
4. Click **Score conversation**.

## How it works

```
Analyze page ──POST /api/analyze──► server.py ──► analyze.py ──► trained model
     ▲                                                              │
     └──────────── scores, turning points, status ◄─────────────────┘
```

`analyze.py` reuses the steps `make export` already runs, applied to one chat: tag cues, score
each turn, find and explain turning points, and set the status. **Nothing uploaded is saved.**

## What changed

| File | Change |
|---|---|
| `inclination_score/src/inclination/analyze.py` | **New.** Scores one chat. |
| `inclination_score/api/server.py` | New endpoints: `POST /api/analyze` (score a chat) and `GET /api/analyze/models` (which models are trained). |
| `inclination_score/config.yaml` | New `analyze:` limits: 2 to 200 messages, at most 2,000 characters each. |
| `web/src/pages/Analyze.jsx` | Uses the real model, asks the conversation type, and says which model scored the chat. |
| `web/src/api/client.js` | Calls the new endpoint. Falls back to the keyword preview only if no model is available. |
| `inclination_score/tests/test_analyze.py`, `test_api.py` | **New.** 22 tests. |
| `requirements.txt`, `pyproject.toml`, READMEs | `httpx` for the API tests, test import path, docs. |

## Things to know

- **Pick the right conversation type.** Each model learned from one dataset and does poorly on
  the others (cross-domain ROC AUC 0.49 to 0.63). The default is the SaaS sales model.
- **Only v1 and v3 can score new chats.** v2 needs a paid Gemini call for every message.
- **Restart `make api` after retraining**, because the server keeps loaded models in memory.
- **If no model is trained** or the server is down, the page falls back to the keyword preview
  and labels the result as a preview.

## Tests

```
cd inclination_score && make test
```

The new tests check that:

- an uploaded chat gets the same scores as the model gives directly
- changing a later message never changes earlier scores
- the response has every field the page needs
- bad input is rejected with a clear message

After `make all`, one more test checks that a real conversation scored as an upload matches its
exported scores exactly.
