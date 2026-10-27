# Weekly Project Report — Week 08

**Team Name:**  
**Date:**  October 21, 2026

## Team Members

| Name | Email | GitHub Username | Role |
|---|---|---|---|
| Taner Bulbul | tbulbul@umd.edu | tbdestroyer | Product Lead / Developer |
| Ayesha Khan | ayekhan@umd.edu | ayekhan12 | Product Idea / DS |
| Namratha Jeetendra | namrath4@umd.edu | namrathajeetendra | |
| Shantanu Ramavat | sramavat@umd.edu | sramavat-24 | Software Engineer |
| Vaibhav Devarapalli | gdevarap@umd.edu | vaibhavdevarapalli | |

## 1. Shipped This Week



## 2. User / Validation Learning



## 3. Metrics Snapshot



## 4. Challenges / Blockers



## 5. Next Week's Goals



## 6. Individual Contributions

### Taner Bulbul

**Contributions:**
I loaded and built the code and run the UI app and used the annotate menu containing annotations for 60 conversations—20 each from Craigslist, P4G, and SaaS. Committed tbulbul1.jsonl file created from the annotate. I also reviewed the latest commits from other team members.
**GitHub Evidence:** commit code: 8cdbb9a18b0bd35cb578ba04d54c084b26da1186
branch: tbulbul_working_branch_1

### Ayesha Khan

**Contributions:** I liaised with the team to schedule our weekly touchpoint so everyone is up to speed on the progress so far, and to decide on the next steps. Scheduled next meeting for Oct 28th. Also shared status updates with those absent from the meeting. 

**GitHub Evidence:**

### Namratha Jeetendra

**Contributions:** I built a new React front end for the project (`web/`) that shows the same pipeline results as the Streamlit dashboard, with a redesigned interface. It has a Pipeline overview with leads grouped by status (Hot, Rising, Stalled, Cooling), a Leads view with the same filters as Streamlit, a Conversation view with the score curve, turning points and explanations, a Script Insights view, and the Annotate tool. I added a small read-only data server (`inclination_score/api/server.py`, run with `make api`) so the React app can read the files in `outputs/` without changing the pipeline or the Streamlit app. I also added an "Analyze new chat" page where a new conversation can be pasted or uploaded; for now it uses placeholder scoring and is ready to connect to a backend endpoint (`POST /api/analyze`, documented in `web/README.md`). I ran the pipeline locally to test the UI against real outputs, opened a pull request for team review, and merged it into main after everyone approved.

**GitHub Evidence:**

### Shantanu Ramavat

**Contributions:**

**GitHub Evidence:**

### Vaibhav Devarapalli

**Contributions:** I built the full-dataset version of the pipeline that I proposed last week and merged it into main. CraigslistBargains now uses the original release from CodaLab instead of the Hugging Face copy, which had no outcome for 2,219 of the 6,682 dialogues. Everything else in Shantanu's pipeline (models, settings, dashboard, the other two datasets) is unchanged.

**What was done:**

- Data: rewrote the Craigslist loader to read the original files. Craigslist goes from 4,462 to 6,331 conversations, and from 95 to 216 no-deals in the test split. The deal rate drops from 85.8% to 77.3%, because the excluded dialogues were mostly no-deals.
- Comparability: the 4,462 dialogues used before come out exactly the same and keep their ids and train/val/test split; the recovered dialogues are split on their own. The annotation sample is unchanged.
- Training and evaluation: reran the pipeline and retrained v1 and v3 on all three datasets. Craigslist test ROC AUC goes from 0.746 to 0.827 (v1) and from 0.880 to 0.908 (v3). Halfway through a conversation it goes from 0.59 to 0.73 (v1) and from 0.63 to 0.74 (v3).
- Caveats: on the same 670 test conversations as before, v1 goes from 0.746 to 0.768, so part of the gain comes from the recovered conversations being easier. Calibration on Craigslist got worse, and SaaS v3 came out at 0.599 against 0.667 with unchanged data (retraining variation).
- Docs and tests: updated the README, PROJECT_EXPLAINER, data notes and report, and the loader tests (81 tests).

**GitHub Evidence:** Commit code: 8be1d79
Pull request #3, merged into main: https://github.com/tbdestroyer/NLP_Project/pull/3

## 7. Lean Canvas Changes (If Any)

**Problem:**

**Customer Segments:**

**Unique Value Proposition:**

**Proposed Solution:**

**Key Metrics:**

**Changes Since Last Week:**
