# Weekly Project Report — Week 05

**Team Name:**  
**Date:** September 29, 2026

## Team Members

| Name | Email | GitHub Username | Role |
|---|---|---|---|
| Taner Bulbul | tbulbul@umd.edu | tbdestroyer | Product Lead / Developer |
| Ayesha Khan | ayekhan@umd.edu | ayekhan12 | |
| Namratha Jeetendra | namrath4@umd.edu | namrathajeetendra | |
| Shantanu Ramavat | sramavat@umd.edu | | |
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
Links to papers used for my background research: 
- **ESD–ERC (2022)** — Emotion shift detection + emotion recognition. [Paper](https://www.sciencedirect.com/science/article/pii/S0950705122004117?)
- **EmoShiftNet (2025)** — Multi-task emotion + emotion-shift detection. [Paper](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1618698/full?)
- **DialogueRNN (2019)** — Models speaker states and conversational history. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4657?)
- **DialogueGCN (2019)** — Graph-based modeling of speaker/context dependencies. [Paper](https://aclanthology.org/D19-1015/?)
- **COSMIC (2020)** — Context + commonsense reasoning for conversational emotion. [Paper](https://aclanthology.org/2020.findings-emnlp.224/?)
- **Shapes of Emotions (2022)** — Explicitly studies emotion shifts in conversation. [Paper](https://aclanthology.org/2022.mmmpie-1.6/?)
- **DAG-ERC (2021)** — Models short- and long-range conversational dependencies. [Paper](https://aclanthology.org/2021.acl-long.123/?)
- **MELD (2019)** — Multi-turn dataset with turn-level emotion and sentiment labels. [Paper](https://aclanthology.org/P19-1050/?) · [Dataset/code](https://github.com/declare-lab/MELD?)

Drafting Parts of Proposal for next week after relevant paper and architecture research: 
Our project develops an early-warning framework for negative sentiment inflection in multi-turn conversational AI. Instead of treating customer sentiment as an independent classification problem for each message, we model how the customer's affect and dialogue state evolve throughout an interaction and ask whether that trajectory predicts an upcoming negative shift.
We build on prior work on Emotion-Flip Reasoning, which identifies utterances responsible for observed changes in emotion, and TRACER, which demonstrates that dialogue failures can be forecast from partial conversational trajectories. We extend these ideas toward a customer-service setting where the goal is to predict a negative emotional transition before it occurs, rather than only recognize or explain it afterward.
Our main contributions are:
 Early sentiment-inflection prediction: We formulate the task as forecasting whether a customer's sentiment will become substantially more negative within a future dialogue window using only the conversation observed so far.
 Trajectory-aware modeling: We combine learned textual representations with temporal signals such as sentiment change, sentiment slope, repeated intents, dialogue-state conflicts, and unresolved interaction patterns, motivated by the dual-stream approach used in TRACER.
 Explainable warning signals: We investigate whether the model can identify the previous utterance or conversational event most strongly associated with an impending or observed sentiment shift, drawing on Emotion-Flip Reasoning and SHARK.
 Customer-service evaluation: We explore transfer to customer-service conversations using the ABCD dataset, with BETOLD providing an additional reference for dialogue breakdown and abandonment-related outcomes.
 Intervention-oriented design: As an extension, predicted risk can be used to trigger clarification, dialogue repair, or human escalation before the conversation fully breaks down, inspired by early-failure forecasting and Detect–Explain–Escalate architectures.
The central hypothesis is that a customer's emotional trajectory contains predictive information before an explicit negative turn occurs, and that combining that trajectory with the semantic content and structure of the dialogue will provide earlier and more actionable warning signals than classifying individual messages in isolation. This is the hypothesis I would make the centerpiece of the project proposal.

**GitHub Evidence:**

### Ayesha Khan

**Contributions:** Discussed concerns on the feasibility and potential challenges of performing a "real-life" evaluation for our chosen topic of identifying "sentiment-switching in the midst of a conversation". Next steps for me are to find related research papers.

**GitHub Evidence:** commit hash: e1f8546

### Namratha Jeetendra

**Contributions:** I took part in our team meeting to define the project's direction and scope. Since we can't evaluate with real users, I proposed an alternative model-based evaluation approach: comparing a baseline model, a fine-tuned model, and a context-aware model on how well they predict sentiment inflection points in customer-support conversations. I also helped outline what the final project will look like so we could lock down its scope, and took part in dividing the remaining tasks across the team.

**GitHub Evidence:**

### Shantanu Ramavat

**Contributions:**

**GitHub Evidence:**

### Vaibhav Devarapalli

**Contributions:**

**GitHub Evidence:**

## 7. Lean Canvas Changes (If Any)

**Problem:**

**Customer Segments:**

**Unique Value Proposition:**

**Proposed Solution:**

**Key Metrics:**

**Changes Since Last Week:**
