# Weekly Project Report — Week 06

**Team Name:**  
**Date:** October 13, 2026

## Team Members

| Name | Email | GitHub Username | Role |
|---|---|---|---|
| Taner Bulbul | tbulbul@umd.edu | tbdestroyer | Product Lead / Developer |
| Ayesha Khan | ayekhan@umd.edu | ayekhan12 | Product Idea / DS |
| Namratha Jeetendra | namrath4@umd.edu | namrathajeetendra | |
| Shantanu Ramavat | sramavat@umd.edu | sramavat-24 | Software Engineer |
| Vaibhav Devarapalli | gdevarap@umd.edu | | |

## 1. Shipped This Week

### Proposed Project Idea:
**Predicting Sentiment Inflection Points in Conversational AI (Chatbots)**

While customer chatbots are widespread, poor conversational flow can lead to user frustration and abandonment. This project models dialogue sentiment trajectories to identify critical inflection points where customer sentiment turns negative. Potential extensions include identifying conversational failures such as intent misclassification or circular routing and determining when escalation to a human agent may be appropriate.

### Project Research / Feasibility

- Reviewed existing research and applications related to conversational sentiment analysis, negative sentiment prediction, and chatbot failure detection.
- Investigated whether similar systems already exist and where there may be opportunities to differentiate our project.
- Considered alternative scopes and variations of the original project idea.
- Investigated potential datasets and methods for evaluating sentiment changes across multi-turn conversations.

## 2. User / Validation Learning



## 3. Metrics Snapshot



## 4. Challenges / Blockers

- Need to determine how novel the proposed project is compared with existing research.
- Need to identify a suitable multi-turn conversational dataset with useful sentiment or dialogue-quality labels.
- Need to narrow the project scope so that it is achievable within the semester.
- Need to decide whether the primary task will be sentiment change detection, early prediction, failure diagnosis, or a combination of these.

## 5. Next Week's Goals

- Finalize the project's primary research question and scope.
- Select an appropriate conversational dataset.
- Define the prediction target and evaluation metrics.
- Establish a simple baseline model for comparison.
- Begin implementing a small proof-of-concept sentiment analysis pipeline.
- Define the minimum viable product and divide initial implementation tasks among team members.

## 6. Individual Contributions

### Taner Bulbul

**Contributions:**
I joined the team meeting and helped discuss the updated project direction and scope. We clarified that the main goal is to compute an inclination score for each stage of a sales conversation, representing how likely the customer is to make a purchase, rather than focusing only on sentiment changes. I also took part in reviewing the first working version of the pipeline, discussing the current results and limitations, and planning next steps around validation, model improvements, and the overall final project structure.
**GitHub Evidence:**

### Ayesha Khan

**Contributions:** Met with the team. Discussed the basic goal is not to measure 'sentiment switch' per se, but primarily to compute the metric 'inclination score' which will be computed per chat, and will measure how likely a customer is to purchase the product. We decided to get the first version running as the next step.

Created a branch from the existing code so I can understand/ experiment with it (ayesha-inclination-score-experimental).

Next step: The next step for me and the team is to understand the code for the first version created by Shantanu Ramavat. And plan on the next steps - potentially for improved accuracy, validation (manual), and user interface.

**GitHub Evidence:** Branch created: 'ayesha-inclination-score-experimental'
commit hash: 3fdcc2c

### Namratha Jeetendra

**Contributions:** I took part in our team meeting to define the project's direction and scope. Since we can't evaluate with real users, I proposed an alternative model-based evaluation approach: comparing a baseline model, a fine-tuned model, and a context-aware model on how well they predict sentiment inflection points in customer-support conversations. I also helped outline what the final project will look like so we could lock down its scope, and took part in dividing the remaining tasks across the team.

**GitHub Evidence:** commit code: df66faa

### Shantanu Ramavat

**Contributions:** I built the first working version of the project's model pipeline and added it to the repository under `inclination_score/`. It scores a sales conversation turn by turn (the probability that the customer ends up buying, using only the turns seen so far), finds turning points in that score, and shows the results in a dashboard.

**What was added:**

- **Data:** loaders for CraigslistBargains, Persuasion for Good and SaaS Sales Conversations (synthetic), with outcome-revealing turns removed and a saved 70/15/15 split by conversation.
- **Models:** a logistic regression baseline on rule-based cues and TF-IDF (v1) and a fine-tuned `distilroberta-base` (v3), both trained on conversation prefixes and calibrated on the validation split.
- **Analysis:** turning-point detection, explanations with a faithfulness check, and agent-move insights.
- **Evaluation:** test ROC AUC of 0.746 (v1) and 0.880 (v3) on Craigslist, 0.616 and 0.605 on Persuasion for Good, 0.778 and 0.667 on SaaS. Early prediction is weak (0.57 to 0.63 at the halfway point). Full results and limitations are in `inclination_score/eval/report.md`.
- **Dashboard and tests:** a Streamlit dashboard (lead list, conversation view, script insights, annotation tool) and 78 pytest tests.

**Still open:** the 60-conversation validation sample has only LLM annotations so far and needs human labels from the team; the Gemini-based features are implemented but not yet run in full.

**GitHub Evidence:**

- `5bf0fbf` Add Inclination Score pipeline and dashboard (66 files in `inclination_score/`)
- `9276549` Add Week 06 report with Shantanu's contributions
- Pull request #1, merged into `main`: https://github.com/tbdestroyer/NLP_Project/pull/1

### Vaibhav Devarapalli

**Contributions:** In the team meeting I proposed using the complete version of the CraigslistBargains dataset. The Hugging Face copy used in the first version has no outcome for 2,219 of the 6,682 dialogues (about a third of the data, including its whole test split), so those had to be excluded. The original release on CodaLab has the deal outcome for every dialogue, which would give us more conversations and many more no-deal examples to train and test on. The team agreed, and I downloaded the original release and started working on a full-data version of the pipeline.

**GitHub Evidence:** commit code: d559374

## 7. Lean Canvas Changes (If Any)

**Problem:**

**Customer Segments:**

**Unique Value Proposition:**

**Proposed Solution:**

**Key Metrics:**

**Changes Since Last Week:**
