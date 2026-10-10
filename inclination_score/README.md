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

```
make setup     # create .venv and install requirements.txt
make all       # data -> train -> export -> eval   (about 10 minutes, no GPU, no API key)
make app       # open the dashboard at http://localhost:8501
make test      # 78 tests
```

`make all` is the four steps below in order. Each can be run on its own. The times are for the
v1 baseline only; once v3 models exist in `outputs/models/`, `export` also scores them and takes
about 40 minutes on the GPU.

| target | what it does | time |
|---|---|---|
| `data` | downloads the three datasets into `data/raw/` (about 205 MB), removes outcome-revealing turns, and writes `data/processed/` with the saved train/val/test split | 1 min |
| `train` | trains the v1 cue model per dataset into `outputs/models/` | 2 min |
| `export` | scores every conversation and writes the dashboard files into `outputs/` | 6 min |
| `eval` | writes `outputs/metrics.json`, `outputs/figures/` and `eval/report.md` | 1 min |

Other targets:

| target | what it does |
|---|---|
| `inspect` | prints the real schema, three example records and outcome statistics of each raw dataset |
| `train-v3` | optional: fine-tunes the transformer per dataset. Needs a GPU and `requirements-ml.txt` (see below). About 2.3 hours on a 4 GB GTX 1650 Ti. Afterwards run `export` (about 40 minutes with v3) and `eval` |
| `llm` | optional, costs money: Gemini cue tags, v2 rubric scores and rationales. Needs `GEMINI_API_KEY` in a `.env` file. About 4 hours at 10 calls per minute and about $3. Every answer is cached, so a stopped run resumes where it left off |
| `export-v2` | adds the Gemini results to the dashboard files without redoing v1 and v3; follow with `eval` |
| `app` | starts the dashboard |
| `test` | runs the tests: loaders, leakage rule, split rule, scoring, exports, evaluation, dashboard pages, annotation, LLM jobs (with a stub, no network) |

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
| CraigslistBargains | real people, negotiation | buyer / seller | the deal was accepted | 4,462 | 85.8% |
| Persuasion for Good | real people, donation | persuadee / persuader | the persuadee's **actual** donation is above $0 | 1,017 | 53.6% |
| SaaS Sales Conversations | **synthetic** (GPT-4) | customer / sales rep | the dataset's `outcome` is 1 | 20,000 of 100,000 | 49.9% |

Splits are 70/15/15 by conversation, stratified by outcome, seed 42, saved in
`data/processed/splits.json`. Turn counts exclude the turns removed to prevent leakage.

| dataset | split | conversations | avg turns | outcome rate |
|---|---|---|---|---|
| craigslist | train / val / test | 3,123 / 669 / 670 | 8.71 / 8.63 / 8.63 | 85.8% in each |
| p4g | train / val / test | 711 / 153 / 153 | 20.56 / 20.71 / 20.56 | 53.6% in each |
| saas | train / val / test | 14,000 / 3,000 / 3,000 | 12.17 / 12.14 / 12.13 | 49.9% in each |

How outcomes and leakage are handled:

- **Craigslist:** outcome = 1 when the dialogue contains an `accept` act. Every `offer`,
  `accept`, `reject` and `quit` turn is dropped, and so is anything typed after an `accept` or
  `quit`. 2,219 of the 6,682 dialogues come without dialogue acts in the available copy, so their
  outcome cannot be recovered and they are excluded. Dialogues that end without any accept,
  reject or quit count as no deal.
- **Persuasion for Good:** the label is the actual donation recorded after the chat, not the
  stated intention. The survey and donation file is used for nothing else.
- **SaaS (synthetic):** a stratified sample of 20,000 conversations. `system` messages and all
  generated annotation columns, including a per-turn "conversion probability", are dropped.

## Results (test split)

| dataset | base rate | model | ROC AUC (95% CI) | AUC at 50% of turns | precision@20 |
|---|---|---|---|---|---|
| CraigslistBargains | 85.8% | v1 | 0.746 (0.698–0.791) | 0.585 | 100% |
| | | v3 | **0.880** (0.847–0.910) | 0.629 | 100% |
| Persuasion for Good | 53.6% | v1 | 0.616 (0.527–0.701) | 0.605 | 70% |
| | | v3 | 0.605 (0.515–0.691) | 0.572 | 80% |
| SaaS (synthetic) | 49.9% | v1 | **0.778** (0.762–0.793) | 0.613 | 95% |
| | | v3 | 0.667 (0.648–0.686) | 0.621 | 90% |

- **Ranking by outcome works, modestly.** Both models beat chance on every dataset. v3 clearly
  helps on Craigslist, ties on Persuasion for Good and is worse than v1 on SaaS.
- **Early prediction is weak.** Halfway through a conversation, every result is between 0.57
  and 0.63.
- **Explanations are faithful.** Removing the trigger turn moves the score 2.5 to 6.2 times more
  than removing a random turn of the same conversation.
- **Turn-level movements do not match the annotator.** Against the LLM annotator, the score's
  direction is right 43% (v1) and 45% (v3) of the time, below the 52% of always answering "same"
  (kappa 0.13 and 0.04). Turning-point precision and recall are between 6% and 14%.
- **No cross-domain transfer.** Off-diagonal ROC AUC is 0.49 to 0.62; the transformer trained
  on synthetic SaaS data is at chance on both real datasets.

Full tables, reliability diagrams and case studies: [eval/report.md](eval/report.md).

## Known limitations

- **Proxy data.** Bargaining and charity donation stand in for sales. Findings may not carry
  over to real sales conversations.
- **Synthetic data.** The SaaS dataset is LLM-generated and many of its conversations end
  without a visible decision. It shows the pipeline works, not that the method works.
- **v3 is not the planned model.** The plan was ModernBERT-base. It does not fit in the 4 GB of
  the GTX 1650 Ti at the input lengths needed, so the spec's fallback, `distilroberta-base`, was
  used: 512 tokens, SaaS trained on 4,000 of 14,000 training conversations, scores for the
  validation and test splits only.
- **No LLM results.** v2 rubric scores, the LLM cue tagger and LLM rationales are built but not
  yet run in full, so nothing here depends on them.
- **No human labels.** The agreement results rest on one LLM annotator, so there is no
  inter-annotator agreement, and an LLM may share blind spots with the models.
- **The score is not reliable turn by turn.** It ranks conversations by how they end, but its
  individual rises, drops and turning points do not match a reader's judgement.
- **Small and imbalanced test sets.** Persuasion for Good has 153 test conversations; Craigslist
  has about 95 negatives in test. Intervals are wide.
- **A third of Craigslist is unused** because its labels are unavailable.
- **Fixed thresholds do not fit every dataset.** With an 86% base rate, nearly every Craigslist
  lead is "Hot" and few have turning points under v1.
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
