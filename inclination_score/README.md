# Inclination Score

A dashboard that helps a sales team **prioritise follow-ups and improve their scripts**. After
every message of a conversation it estimates the probability that the customer ends up buying,
finds the turning points where that probability rises or drops sharply, and explains what
triggered them.

- Plain-language explanation for the group and professor: [PROJECT_EXPLAINER.md](PROJECT_EXPLAINER.md)
- Results: [eval/report.md](eval/report.md)
- Data notes: [data/README.md](data/README.md)

There is no public dataset of real sales chats with purchase labels, so the project uses
**proxies**: price negotiation and charity donation between real people, plus one synthetic
sales dataset. Read the [limitations](#known-limitations) before quoting any number.

## Run it from scratch

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/). With GNU make, use `make <target>`.
On Windows without make, use `.\make.ps1 <target>`; the targets are the same.

First save the three CraigslistBargains files by hand (CodaLab cannot be downloaded from
automatically): open the [CodaLab worksheet](https://worksheets.codalab.org/worksheets/0x453913e76b65495d8b9730d41c7e0a0c/),
open `train.json`, `dev.json` and `test.json`, and save each under that name in
`data/raw/craigslist/`. Details: [data/README.md](data/README.md).

```
make setup     # create .venv and install requirements.txt
make all       # data -> train -> export -> eval   (about 10 minutes, no GPU, no API key)
make app       # open the dashboard at http://localhost:8501
make test      # 106 tests
```

`make all` is the four steps below in order. Each can be run on its own. The times are for the
v1 baseline only; once v3 models exist in `outputs/models/`, `export` also scores them and takes
about 40 minutes on the GPU.

| target | what it does | time |
|---|---|---|
| `data` | checks the Craigslist files, downloads the other two datasets into `data/raw/`, removes outcome-revealing turns, and writes `data/processed/` with the saved train/val/test split | 1 min |
| `train` | trains the v1 cue model per dataset into `outputs/models/` | 2 min |
| `export` | scores every conversation and writes the dashboard files into `outputs/` | 6 min |
| `eval` | writes `outputs/metrics.json`, `outputs/figures/` and `eval/report.md` | 1 min |

Other targets:

| target | what it does |
|---|---|
| `inspect` | prints the real schema, three example records and outcome statistics of each raw dataset |
| `train-v3` | optional: fine-tunes the transformer per dataset. Needs a GPU and `requirements-ml.txt` (see below). About 2.3 hours on a 4 GB GTX 1650 Ti, 1.4 hours on an 8 GB RTX 4070 Laptop GPU. Afterwards run `export` (about 40 minutes with v3) and `eval` |
| `llm` | optional, costs money: Gemini cue tags, v2 rubric scores and rationales. Needs `GEMINI_API_KEY` in a `.env` file. About 4 hours at 10 calls per minute and about $3. Every answer is cached, so a stopped run resumes where it left off |
| `export-v2` | adds the Gemini results to the dashboard files without redoing v1 and v3; follow with `eval` |
| `app` | starts the dashboard |
| `test` | runs the tests: loaders, leakage rule, split rule, scoring, exports, evaluation, dashboard pages, annotation, LLM jobs (with a stub, no network), new-chat analysis and its API |

To add the transformer (v3):

```
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/Scripts/python.exe -r requirements-ml.txt
make train-v3
make export
make eval
```

Everything downloaded stays inside the project folder: raw data in `data/raw/`, Hugging Face
files in `.cache/huggingface/`, `uv` packages in `.cache/uv/`. All settings (paths, seed,
thresholds, model names, sample sizes) are in [config.yaml](config.yaml).

**Gemini features are built but have not been run in full.** The LLM cue tagger, the v2 rubric
scorer and the LLM rationales are implemented and tested, and the dashboard and report pick
their results up when present. The full run was paused after about a quarter of the cue tagging,
so no Gemini results are reported. The pipeline does not need an API key: cues come from rules
and rationales from templates.

## The dashboard

The sidebar picks the dataset, model and split (test by default) and holds the "Show true
outcomes (evaluation mode)" switch. It is off by default; in product mode outcomes are never
shown. The synthetic dataset carries a badge on every page.

| page | what it shows |
|---|---|
| **Home** | what the product does, conversations scored, average score, hot leads, and the test ROC AUC of the selected model |
| **Lead List** | conversations ranked by final score with a sparkline of the curve, a status (Hot, Rising, Cooling, Stalled) and its reason. Filter by status, score range, minimum turns, or text in the title or messages. Select a row, then "View conversation". In evaluation mode: an outcome column and how many of the top 20 converted |
| **Conversation View** | the score curve with turning-point triangles next to the transcript with cue chips. Pick a turning point, or click a triangle, for its explanation card: trigger message, preceding agent message, cues, size of the change and faithfulness. The replay slider shows the conversation as it stood at any turn. Curves of other models can be overlaid, and the turning-point method can be switched |
| **Script Insights** | mean change in the customer's score on their next turn after each agent move, with 95% intervals, by position in the conversation, plus three example snippets per move. These are associations, not proven causes |
| **Annotate** | internal labelling tool, described below |

The first four pages only read files in `outputs/`; they never train a model or call an LLM. If
a file is missing, the page says which one and which command creates it. The colour theme (cream
with light green) is set in `.streamlit/config.toml` and `app/common.py`.

**Scoring a new chat.** The React app in `web/` has an Analyze page: upload or paste any chat,
mark who is the customer, and a trained model (v1 by default) scores it like the datasets, with
turning points, explanations and a status. It runs through `make api`
(`POST /api/analyze`, code in `src/inclination/analyze.py`) and stores nothing. Each model knows
one kind of conversation only, so the page asks which kind it is. Upload many chats at once
(`POST /api/analyze/batch`) to get them ranked as a lead list with script insights over the batch.
Details: [web/README.md](../web/README.md).

### Annotate (internal tool)

Builds the labelled validation set. It shows 60 fixed test-split conversations (20 per dataset,
listed in `annotation/sample.json`) with model scores, cues and outcomes hidden. For each
customer turn the annotator picks more / same / less inclined, and can mark any turn as a turning
point. Labels go to `annotation/labels/<annotator>.jsonl`; any number of annotators can work side
by side. Instructions and three worked examples: [annotation/guidelines.md](annotation/guidelines.md).

**The only labels so far are LLM annotations, not human ones.** `annotation/labels/auto_llm.jsonl`
was produced by an LLM from the transcripts alone (no outcomes, scores or cues), following the
guidelines. Its readable source is `annotation/llm_labels.txt`; re-import it with
`python -m inclination.annotation annotation/llm_labels.txt auto_llm`. Results computed
against it must be described as agreement with an LLM annotator. Once human label files exist,
`make eval` adds human agreement and agreement between annotators to the report.

## How it works

1. **Data.** Three datasets are converted to one format: a turn table (`dataset`,
   `conversation_id`, `turn`, `role`, `text`) and a conversation table with the outcome and split.
2. **Leakage removal.** Anything that reveals the outcome is removed before scoring. A score at
   turn *k* uses turns 0..*k* only; a test alters later turns and checks that earlier scores do
   not move.
3. **Cues.** A rule-based tagger marks 12 customer cues (for example price objection,
   commitment) and 10 agent moves (for example gives discount, pushes close).
4. **Scoring.** Every conversation is cut into prefixes, each labelled with the final outcome.
   - **v1:** logistic regression on cue counts, agent-move counts and TF-IDF of the customer's text.
   - **v3:** fine-tuned transformer on `[CUSTOMER] … [AGENT] …` text, most recent turns kept.
   - Scores are calibrated on the validation split (isotonic; Platt when validation is small).
5. **Turning points.** Method A: the score changes by at least 0.15 in one turn. Method B: PELT
   change-point detection on the score curve.
6. **Explanations.** Trigger message, cues, a template sentence, the largest feature contributions
   (v1), and a faithfulness check that removes the trigger and re-scores.
7. **Script insights.** For each agent move, the change in score on the customer's next turn,
   with a bootstrap confidence interval.

## Datasets

| dataset | kind | customer / agent | outcome = 1 when | conversations | outcome rate |
|---|---|---|---|---|---|
| CraigslistBargains | real people, negotiation | buyer / seller | the deal was accepted | 6,331 | 77.3% |
| Persuasion for Good | real people, donation | persuadee / persuader | the persuadee's **actual** donation is above $0 | 1,017 | 53.6% |
| SaaS Sales Conversations | **synthetic** (GPT-4) | customer / sales rep | the dataset's `outcome` is 1 | 20,000 of 100,000 | 49.9% |

Splits are 70/15/15 by conversation, stratified by outcome, seed 42, saved in
`data/processed/splits.json`. Turn counts exclude the turns removed to prevent leakage.

| dataset | split | conversations | avg turns | outcome rate |
|---|---|---|---|---|
| craigslist | train / val / test | 4,431 / 949 / 951 | 7.77 / 7.71 / 7.72 | 77.3% in each |
| p4g | train / val / test | 711 / 153 / 153 | 20.56 / 20.71 / 20.56 | 53.6% in each |
| saas | train / val / test | 14,000 / 3,000 / 3,000 | 12.17 / 12.14 / 12.13 | 49.9% in each |

How outcomes and leakage are handled:

- **Craigslist:** the original release, with the dataset's own outcome (`reward`) for every
  dialogue. Every `offer`, `accept`, `reject` and `quit` event is dropped, and so is anything
  typed after an `accept` or `quit`. Dialogues that end without any accept, reject or quit count
  as no deal. The first version used the Hugging Face copy, where 2,219 of the 6,682 dialogues had
  no outcome and were excluded; those are now included and split on their own, so the 4,462 used
  before keep their train/val/test assignment.
- **Persuasion for Good:** the label is the actual donation recorded after the chat, not the
  stated intention. The survey and donation file is used for nothing else.
- **SaaS (synthetic):** a stratified sample of 20,000 conversations. `system` messages and all
  generated annotation columns, including a per-turn "conversion probability", are dropped.

## Results (test split)

| dataset | base rate | model | ROC AUC (95% CI) | AUC at 50% of turns | precision@20 |
|---|---|---|---|---|---|
| CraigslistBargains | 77.3% | v1 | 0.827 (0.794–0.859) | 0.732 | 100% |
| | | v3 | **0.908** (0.887–0.929) | 0.741 | 100% |
| Persuasion for Good | 53.6% | v1 | 0.616 (0.527–0.701) | 0.605 | 70% |
| | | v3 | 0.609 (0.514–0.693) | 0.501 | 80% |
| SaaS (synthetic) | 49.9% | v1 | **0.778** (0.762–0.793) | 0.613 | 95% |
| | | v3 | 0.599 (0.580–0.617) | 0.571 | 80% |

- **Ranking by outcome works, modestly.** Both models beat chance on every dataset. v3 clearly
  helps on Craigslist, ties on Persuasion for Good and is worse than v1 on SaaS.
- **The full Craigslist release helps.** Against the first version (Hugging Face copy, 4,462
  dialogues), test ROC AUC goes from 0.746 to 0.827 (v1) and from 0.880 to 0.908 (v3). On the
  670 test conversations both versions share, v1 goes from 0.746 to 0.768, so part of the gain
  comes from the recovered conversations being easier to predict.
- **Early prediction is weak except on Craigslist.** Halfway through a conversation, Craigslist
  reaches 0.73 (v1) and 0.74 (v3), against 0.59 and 0.63 before; the other datasets stay between
  0.50 and 0.61.
- **Explanations are faithful.** Removing the trigger turn moves the score 2.8 to 6.6 times more
  than removing a random turn of the same conversation.
- **Turn-level movements do not match the annotator.** Against the LLM annotator, the score's
  direction is right 46% (v1) and 48% (v3) of the time, below the 52% of always answering "same"
  (kappa 0.17 and 0.08). Turning-point precision and recall are between 12% and 24%.
- **No cross-domain transfer.** Off-diagonal ROC AUC is 0.49 to 0.63; the transformer trained
  on synthetic SaaS data is close to chance on both real datasets.
- **v3 was retrained for all three datasets** on an RTX 4070 Laptop GPU (8 GB) with the same
  settings, about 1.4 hours in total. SaaS v3 came out lower than in the first version (0.599
  against 0.667) although its data did not change: its validation AUC stayed near 0.55 and
  training stopped early. This is most likely run-to-run variation of the transformer, not an
  effect of the Craigslist change. Persuasion for Good v3 moved from 0.605 to 0.609.

Full tables, reliability diagrams and case studies: [eval/report.md](eval/report.md).

## Known limitations

- **Proxy data.** Bargaining and charity donation stand in for sales. Findings may not carry
  over to real sales conversations.
- **Synthetic data.** The SaaS dataset is LLM-generated and many of its conversations end
  without a visible decision. It shows the pipeline works, not that the method works.
- **v3 is not the planned model.** The plan was ModernBERT-base. It did not fit in the 4 GB GPU
  used for the first version (a GTX 1650 Ti) at the input lengths needed, so the spec's fallback,
  `distilroberta-base`, was used: 512 tokens, SaaS trained on 4,000 of 14,000 training
  conversations, scores for the validation and test splits only. The full-data rerun used an 8 GB
  RTX 4070 Laptop GPU but kept the same model and settings, so the two versions can be compared.
- **No LLM results.** v2 rubric scores, the LLM cue tagger and LLM rationales are built but not
  yet run in full, so nothing here depends on them.
- **No human labels.** The agreement results rest on one LLM annotator, so there is no
  inter-annotator agreement, and an LLM may share blind spots with the models.
- **The score is not reliable turn by turn.** It ranks conversations by how they end, but its
  individual rises, drops and turning points do not match a reader's judgement.
- **Small and imbalanced test sets.** Persuasion for Good has 153 test conversations, so its
  intervals are wide. Craigslist has 216 negatives in test (95 in the first version).
- **Fixed thresholds do not fit every dataset.** Even with the full release (77% base rate),
  about 96% of Craigslist test leads are "Hot" under v1.
- **Keyword-based cues.** The tagger misses paraphrases and several cues are rare.
- **Insights are correlational,** and measured with models that are themselves weak.
- **Nothing was tuned.** Hyperparameters are defaults; v1 overfits.

## Repository layout

```
config.yaml            all settings
Makefile, make.ps1     the commands above
src/inclination/
  load/                download, inspection, one loader per dataset, standardise, split
  cues/                taxonomy, rule-based tagger, Gemini tagger
  scoring/             cue_model (v1), llm_rubric (v2), transformer (v3), calibrate
  llm.py, llm_jobs.py  cached Gemini client and the three LLM jobs
  turning_points.py    methods A and B
  explain.py           rationales and trigger-removal variants
  insights.py          agent-move effects
  annotation.py        labelling sample and label files
  train.py, export.py  pipeline entry points
app/                   Streamlit dashboard (Home, three views, Annotate)
eval/                  evaluate.py and the generated report.md
annotation/            guidelines, sample, label files
tests/                 pytest suite
data/, outputs/        generated; not meant to be edited by hand (only outputs/figures/ is in git)
```
