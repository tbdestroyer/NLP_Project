# Inclination · Sales workspace

A React and Vite sales workspace for Inclination Score. It reads the same exported pipeline results
as the Streamlit dashboard (`inclination_score/app/`) and includes a browser-only analysis preview.
Analyze scores an uploaded chat with a trained model through the same data server.

| Screen | What it shows |
|---|---|
| Pipeline | Live summary metrics and a temperature board of ranked leads |
| Leads | Board or table views, status and score filters, minimum turns, title/message search, evaluation metrics |
| Conversation | Right-side conversation view with a Recharts score curve, replay, available method/cue controls, overlays, evidence and transcript |
| Insights | Helpful/harmful agent moves, timing tabs, confidence ranges, data table and transcript examples |
| Annotate | Focused sample labelling queue with keyboard shortcuts; writes to the same `annotation/labels/<name>.jsonl` files as Streamlit |
| Analyze | Upload or paste a chat (text, CSV or JSON), mark the speakers, pick the kind of conversation, and see its score curve, turning points and status from a trained model |

## Run it

First-time setup and real pipeline exports (from the repository root):

```bash
cd inclination_score
make setup        # first time only
make all          # downloads data, trains v1, exports results and evaluates (can take a while)
```

Then use two terminals to start the services:

```bash
# terminal 1: from the repository root
cd inclination_score
make api

# terminal 2: from the repository root
cd web
npm install       # first time only; installs the UI and component libraries
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). Keep both terminals
running while using the dashboard.

## Where the data comes from

| Screen | Data |
|---|---|
| Overview, Lead List, Conversation View, Script Insights | **Real** pipeline results from the same Parquet/JSON files used by Streamlit, including `insight_examples.parquet`. |
| Annotate | The fixed `annotation/sample.json`, transcripts from `turns.parquet`, and append-only `annotation/labels/<name>.jsonl` files. |
| Analyze new chat | A **trained model** in `outputs/models/` (`make train`), through `POST /api/analyze`. Nothing is stored. If the server is down or no model is trained, the page falls back to keyword rules in the browser (`src/data/scoring.js`) and labels the result as a preview. |

`outputs/` is not in git. The API needs `leads.parquet`, `scores.parquet`, `turns.parquet`,
`cues.parquet`, `turning_points.parquet`, `insights.parquet`, and `insight_examples.parquet`.
`metrics.json` is optional. `make all` trains v1 only; v3 requires the separate `make train-v3`
target and a GPU. If the server is stopped or an output is missing, the UI explains how to recover.
The model picker only lists models present in the exported leads for the selected dataset.

The data server is `inclination_score/api/server.py`. In development, Vite forwards every
`/api/...` request to it (`vite.config.js`), so there's nothing to configure.

The web UI is styled with Tailwind CSS and local shadcn-style components backed by Radix UI.
Charts use Recharts, transitions use Framer Motion, and icons use lucide-react. The theme toggle
switches between light and dark mode and remembers the choice in this browser.

### Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{ ok, outputs, missing[] }` |
| GET | `/api/options` | `{ dataset: [models] }` that were exported |
| GET | `/api/leads?dataset=&model=&split=` | rows of `leads.parquet`, plus `curve` (scores in turn order) |
| GET | `/api/search?dataset=&q=` | ids of conversations whose messages contain `q` |
| GET | `/api/conversations/{id}?model=&method=&tagger=&models=` | the lead row plus turns, selected cues, selected model scores, overlays, and available turning-point methods |
| GET | `/api/insights?dataset=&model=` | rows of `insights.parquet` |
| GET | `/api/insight-examples?dataset=&model=&move=&position=` | paired transcript turns, rule cues and score changes from `insight_examples.parquet` |
| GET | `/api/metrics?dataset=&model=` | `{ roc_auc, roc_auc_ci }` from `metrics.json`, or `null` |
| GET | `/api/annotations?annotator=&dataset=` | fixed annotation queue and latest saved labels |
| POST | `/api/annotations` | validates and appends labels to `annotation/labels/<name>.jsonl` |
| GET | `/api/analyze/models` | `{ [dataset]: [models] }` trained in `outputs/models/`; works before `make export` |
| POST | `/api/analyze` | scores a new chat; same shape as `/api/conversations/{id}` without an outcome |

### Analyze new chat

`POST /api/analyze` takes `{ dataset, model, title, method?, turns: [{ role: "agent" | "customer", speaker, text }] }`.
`inclination_score/src/inclination/analyze.py` runs the export's steps on that one chat: rule cue
tagging, a calibrated score after every turn, turning points with explanations and faithfulness,
and the lead status. v1 and v3 can score new chats; v2 cannot (it needs a paid LLM call per turn).
Limits (messages, characters per message) are under `analyze:` in `config.yaml`. Models are loaded
once, so restart `make api` after retraining.

Every model is trained on one dataset and transfers poorly to others (cross-domain ROC AUC
0.49 to 0.63), so the page asks what kind of conversation it is, defaults to the SaaS model and
says which model scored the chat.

## Where things live

```
src/
  api/client.js              the only place that fetches data (all calls go to /api)
  data/taxonomy.js           cue names and phrases (mirror of cues/taxonomy.py)
  data/scoring.js            fallback scorer for Analyze when no trained model is available
  components/ConversationDetail.jsx   shared transcript, Recharts curve and turning points
  components/ui/primitives.jsx        local accessible shadcn-style UI primitives
  pages/                     one file per screen
  styles.css                 Tailwind entry point and global theme tokens
tailwind.config.js           Tailwind theme and content paths
postcss.config.js            Tailwind and Autoprefixer setup
```

If you change cue names in `cues/taxonomy.py` or thresholds in `config.yaml`, update
`src/data/taxonomy.js` and `src/data/scoring.js` to match.
