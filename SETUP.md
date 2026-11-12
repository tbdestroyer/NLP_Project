# Setup: run Inclination Score on your laptop

This guide takes you from a fresh laptop to the running web app. The steps are written for
Windows (PowerShell), with notes for Mac and Linux. For background on what the project does, see
[inclination_score/README.md](inclination_score/README.md).

## 1. Install the tools (one time)

| Tool | Why | How to get it |
|---|---|---|
| **Git** | to clone the repository | <https://git-scm.com> |
| **uv** | creates the Python 3.11 environment (it downloads Python 3.11 for you if needed) | Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"`<br>Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Node.js 18 or newer** | runs the web app | <https://nodejs.org> (LTS version) |
| NVIDIA GPU or Apple silicon Mac (optional) | only for the v3 transformer and for Live mode's speech-to-text | — |

Close and reopen the terminal, then check that each tool works:

```powershell
git --version
uv --version
node --version
```

## 2. Get the code

```powershell
git clone https://github.com/tbdestroyer/NLP_Project.git
cd NLP_Project\inclination_score
```

On Mac/Linux the path uses forward slashes: `cd NLP_Project/inclination_score`.

Unless a step says otherwise, run every command from `inclination_score\`.

- **Windows:** `.\make.ps1 <target>`
- **Mac/Linux:** `make <target>`

The targets are the same on both.

## 3. Create the Python environment

```powershell
.\make.ps1 setup
```

This creates `.venv` and installs `requirements.txt`. It takes a few minutes.

## 4. Get the model outputs (choose A or B)

The trained models and scored results live in `inclination_score/outputs/`, which is **not in
git**. Get them one of two ways.

### Option A: use the team's trained models (fastest, includes v3)

1. Ask the team (Vaibhav trained the models) for the `outputs` zip.
2. Unzip it into `inclination_score\outputs\`. The folder should then contain `leads.parquet`,
   `scores.parquet`, `metrics.json`, `models\` and so on.
3. Install the scikit-learn version the models were saved with, or the v1 models may not load:

   ```powershell
   uv pip install --python .venv\Scripts\python.exe scikit-learn==1.9.1
   ```

   On Mac/Linux the environment's Python is `.venv/bin/python`:

   ```bash
   uv pip install --python .venv/bin/python scikit-learn==1.9.1
   ```

4. **Don't run `train`, `train-v3`, `export` or `all` after unzipping.** They overwrite the
   team's results.

### Option B: build everything yourself (about 10 minutes, CPU only, v1 models only)

1. Download the Craigslist data by hand. CodaLab can't be downloaded from automatically.
   - Open the [CodaLab worksheet](https://worksheets.codalab.org/worksheets/0x453913e76b65495d8b9730d41c7e0a0c/).
   - Save `train.json`, `dev.json` and `test.json` into `inclination_score\data\raw\craigslist\`.
     More detail is in [inclination_score/data/README.md](inclination_score/data/README.md).
2. Run the full pipeline:

   ```powershell
   .\make.ps1 all
   ```

   This downloads the other two datasets, cleans the data, trains the v1 models, scores every
   conversation and writes the evaluation.

## 5. Optional: GPU extras (v3 transformer and Live speech-to-text)

Skip this step if you only need the v1 models, the Analyze page, or Live mode with typing or the
demo.

```powershell
$env:UV_CACHE_DIR = ".cache\uv"
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe -r requirements-ml.txt
```

**Mac with Apple silicon:** skip the CUDA index. The standard PyTorch build is enough, and v3
trains and scores on the Apple GPU (`mps`):

```bash
uv pip install --python .venv/bin/python -r requirements-ml.txt
```

Live mode's speech-to-text uses only an NVIDIA GPU. On a Mac it runs Whisper on the CPU, which
works but is slower.

- **If you used option A** and want to use the team's v3 models, match the versions they were
  saved with:

  ```powershell
  uv pip install --python .venv\Scripts\python.exe "torch==2.14.1" "transformers==5.20.2" --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
  ```

  On a Mac:

  ```bash
  uv pip install --python .venv/bin/python "torch==2.14.1" "transformers==5.20.2"
  ```

- **If you used option B** and want v3 as well, train it, then re-export and re-evaluate. This
  takes about 1.5 to 2.5 hours on a laptop GPU.

  ```powershell
  .\make.ps1 train-v3
  .\make.ps1 export
  .\make.ps1 eval
  ```

## 6. Start the app (two terminals)

**Terminal 1: data server** (API on port 8000)

```powershell
cd NLP_Project\inclination_score
.\make.ps1 api
```

**Terminal 2: web app**

```powershell
cd NLP_Project\web
npm install      # first time only
npm run dev
```

Open <http://localhost:5173> in Chrome or Edge, and keep both terminals open while you use the
app.

| Page | What it does |
|---|---|
| Pipeline, Leads, Conversation, Insights | results on the three datasets |
| **Analyze** | upload or paste one chat, or many, and score them with a trained model |
| **Live** (`#/live`) | scores and coaches a sales call as it happens. Without a GPU, use "Type, or play a demo". The first time speech-to-text is used, it downloads Whisper (about 1 GB) |

More detail: [web/README.md](web/README.md), [inclination_score/ANALYZE_README.md](inclination_score/ANALYZE_README.md)
and [inclination_score/LIVE_README.md](inclination_score/LIVE_README.md).

## 7. Optional extras

```powershell
.\make.ps1 app    # older Streamlit dashboard at http://localhost:8501
.\make.ps1 test   # run the test suite
```

## Common problems

| Problem | Fix |
|---|---|
| The page says the server is down or files are missing | Terminal 1 isn't running, or `outputs\` is empty (go back to step 4) |
| Vite serves 404s | Run plain `npm run dev`. In PowerShell, adding `-- --port ...` breaks it |
| `make.ps1` is blocked ("running scripts is disabled") | Run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then try again |
| The microphone doesn't work on Live | Browsers allow the microphone only on `localhost` or HTTPS. Use Chrome or Edge |
| A v1 model fails to load or warns about versions | Install `scikit-learn==1.9.1` (step 4A) |
| The app still shows old models after retraining | Restart terminal 1; the server loads models only once |
