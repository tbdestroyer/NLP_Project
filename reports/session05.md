# Weekly Project Report — Week 05

**Team Name:**  
**Date:** September 29, 2026

## Team Members

| Name | Email | GitHub Username | Role |
|---|---|---|---|
| Taner Bulbul | tbulbul@umd.edu | tbdestroyer | Product Lead / Developer |
| Ayesha Khan | ayekhan@umd.edu | ayekhan12 | |
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
Updated project scope and links to better match objective after meeting tonight: 

Our project focuses on early, trajectory-based purchase prediction in sales conversations. Rather than classifying each message independently, we estimate a turn-by-turn Inclination Score representing the customer’s likelihood of eventually purchasing, then detect important increases or drops and analyze what conversational events may have caused them. The most relevant prior work now includes SalesRLAgent for real-time sales conversion prediction, SalesLLM for buying-intent evaluation in sales dialogue, TRACER for forecasting outcomes from partial dialogue trajectories, DialogueRNN for modeling conversational history, and Emotion-Flip Reasoning for identifying utterances associated with behavioral or emotional shifts. For data, the strongest options are SaaS Sales Conversations as the main conversion dataset, Kapibala Sales Dialogues for turn-level purchase-intent validation, and optionally CraigslistBargain or PersuasionForGood for external validation. This better matches the current proposal than the earlier emotion-recognition-heavy literature.   
Links:
- SalesRLAgent: https://arxiv.org/abs/2503.23303
- SalesLLM: https://arxiv.org/abs/2604.07054
- TRACER: https://arxiv.org/abs/2607.03974
- DialogueRNN: https://ojs.aaai.org/index.php/AAAI/article/view/4657
- Emotion-Flip Reasoning: https://arxiv.org/abs/2306.13959
- SaaS Sales Conversations: https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations
- Kapibala Sales Dialogues: https://huggingface.co/datasets/kapibala-ai/kapibala-sales-dialogues
- CraigslistBargain: https://huggingface.co/datasets/stanfordnlp/craigslist_bargains
- PersuasionForGood: https://convokit.cornell.edu/documentation/persuasionforgood.html

**GitHub Evidence:**

### Ayesha Khan

**Contributions:** Discussed concerns on the feasibility and potential challenges of performing a "real-life" evaluation for our chosen topic of identifying "sentiment-switching in the midst of a conversation". Next steps for me are to find related research papers.

**GitHub Evidence:** commit hash: e1f8546

### Namratha Jeetendra

**Contributions:** I took part in our team meeting to define the project's direction and scope. Since we can't evaluate with real users, I proposed an alternative model-based evaluation approach: comparing a baseline model, a fine-tuned model, and a context-aware model on how well they predict sentiment inflection points in customer-support conversations. I also helped outline what the final project will look like so we could lock down its scope, and took part in dividing the remaining tasks across the team.

**GitHub Evidence:** commit code: df66faa

### Shantanu Ramavat

**Contributions:** I researched additional public datasets to complement the ones my teammate found, focusing on our project's two needs: labeled multi-turn sentiment or emotion data, and data that reflects a customer-service setting. During the team discussion, we agreed to use labeled datasets, so I evaluated each candidate on whether it had turn-level labels, whether the dialogues were multi-turn, and how closely the domain matched customer-service chatbots.

**Dataset Research:**

**ABCD (Action-Based Conversations Dataset):** Real customer-agent dialogues with annotated intents and agent actions. Supports our extensions on conversational failures and escalation. https://github.com/asappresearch/abcd
**Dialogue Breakdown Detection Challenge (DBDC3/DBDC4):** Chatbot conversations with turn-level breakdown labels, which can help us identify where a conversation goes wrong. https://sites.google.com/site/dialoguebreakdowndetection4/datasets?authuser=0
**EmoryNLP:** Multi-party dialogues with utterance-level emotion labels, adding more labeled sentiment trajectories beyond MELD. https://github.com/emorynlp/emotion-detection

**GitHub Evidence:** a119ca4

### Vaibhav Devarapalli

**Contributions:** I researched potential public datasets that could be used for our project. During the team discussion, we considered whether to use labeled or unlabeled conversational datasets and agreed to move forward with labeled datasets for the project. I then researched and looked into several conversational datasets to identify which ones fit our project's focus on sentiment changes across multi-turn conversations.

**Dataset Research:**

- **MELD** — Multi-turn conversations with sentiment and emotion labels.
  https://github.com/declare-lab/MELD

- **DailyDialog** — Multi-turn everyday conversations with emotion and dialogue-act annotations.
  https://github.com/dialoguesystems/dialogue-datasets/tree/master/dailyDialog

- **EmotionLines** — Multi-party conversations with turn-level emotion annotations.
  https://arxiv.org/abs/1802.08379

**GitHub Evidence:** 54aa6fe

## 7. Lean Canvas Changes (If Any)

**Problem:**

**Customer Segments:**

**Unique Value Proposition:**

**Proposed Solution:**

**Key Metrics:**

**Changes Since Last Week:**
