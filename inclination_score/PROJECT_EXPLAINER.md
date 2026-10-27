# Inclination Score: project explainer

*For the project group and the professor. Last updated 2026-10-21, after switching
CraigslistBargains to its full original release and rerunning both models. The Gemini LLM
features are implemented, but their full run was paused part-way, so no Gemini results are
reported here.*

## 1. The project in one paragraph

Inclination Score reads a sales conversation one message at a time and, after every message,
estimates the probability that the customer ends up buying. Plotting those probabilities gives a
curve for each conversation. We look for the places where the curve jumps or falls sharply
("turning points"), explain what was said there, and summarise across all conversations which
things the salesperson does tend to come before a rise or a drop. The results are shown in a
dashboard meant for a sales manager.

**Where we are now:** the whole pipeline runs end to end with two models: a baseline (v1) and a
fine-tuned transformer (v3). Both rank buyers above non-buyers better than chance on all three
datasets, but both are weak early in the conversation except on Craigslist. The transformer is
clearly better on Craigslist (test ROC AUC 0.91 against 0.83), no better on Persuasion for Good
(0.61 against 0.62) and worse on the synthetic SaaS data (0.60 against 0.78). It is a smaller transformer than
planned, because of GPU memory (section 8). The dashboard shows both models. The labelling
tool is ready, but **nobody has labelled anything yet: that is the group's next job**
(section 8a). The LLM features are coded but have not been run in full.

**The full evaluation is done, and it gives a mixed picture** (section 8). The good news: the
score ranks buyers above non-buyers, and the explanations point at messages that really drive
the score. The bad news: the score's turn-by-turn movements and turning points do not agree
with an annotator's judgement, models do not transfer between datasets, and early prediction is
weak outside Craigslist. The honest summary is "useful for ranking leads at the end of a
conversation; not yet reliable as a live, message-by-message reading".

## 2. Why this is worth doing

A sales team has more open conversations than it can follow up on. Two questions matter to them:

1. **Whom should we contact first?** A ranking of conversations by likelihood of buying answers
   this. It is the main purpose of the project and the main thing we evaluate.
2. **What in our script helps or hurts?** Turning points and their explanations answer this for
   one conversation; the script insights view answers it across all of them.

A single end-of-conversation prediction would answer neither well. A per-message score shows
*when* a customer was won or lost, and it can be used while the conversation is still going.

## 3. The task, stated precisely

For a conversation with turns 0..n, the score after turn *k* is

> score_k = P(customer buys | turns 0..k)

Two rules follow from this definition and shape the whole design:

- The score at turn *k* may only use turns up to *k*. It must not change if later turns change.
- Nothing that reveals the outcome may be in the input. The label is used for training and
  evaluation only.

**How one label per conversation becomes a per-turn score.** We only know how each conversation
ended. So we cut every conversation into prefixes (turns 0..0, 0..1, 0..2, and so on) and give
each prefix the conversation's final outcome as its label. A classifier trained on prefixes
learns to predict the ending from a partial conversation. Each conversation's prefixes are
weighted to sum to 1, so long conversations do not dominate training.

## 4. Data

There is no public dataset of real sales chats with purchase labels. We use two real-human
datasets where one person tries to get another to agree to something, plus one synthetic sales
dataset.

| dataset | what it is | customer / agent | "bought" means | conversations used | positive rate |
|---|---|---|---|---|---|
| CraigslistBargains | real people bargaining over a Craigslist listing | buyer / seller | the deal was accepted | 6,331 | 77.3% |
| Persuasion for Good | real people; one persuades the other to donate to a charity | persuadee / persuader | the persuadee actually donated more than $0 | 1,017 | 53.6% |
| SaaS Sales Conversations | **synthetic**, generated with GPT-4 | customer / sales rep | the dataset's `outcome` field is 1 | 20,000 (sampled from 100,000) | 49.9% |

Average length is 7.8, 20.6 and 12.2 turns respectively. In total: 27,348 conversations and
313,220 turns.

**Splits.** 70% train, 15% validation, 15% test, made per dataset, stratified by outcome, with a
fixed seed (42). The split is by whole conversation, so all prefixes of a conversation are in the
same split. The assignment is saved to a file and reused by every later step.

| dataset | train | val | test |
|---|---|---|---|
| craigslist | 4,431 | 949 | 951 |
| p4g | 711 | 153 | 153 |
| saas | 14,000 | 3,000 | 3,000 |

### What we found in the data, and what we did about it

These are the decisions most likely to be asked about.

**CraigslistBargains**
- *We use the original release, which has a label for every dialogue.* The first version used
  the copy on Hugging Face, which has no dialogue-act annotations for 2,219 of 6,682 dialogues,
  including its whole test split; there "accept" and "reject" both appear as empty messages, so
  those dialogues were excluded. The original files (on CodaLab) have to be saved by hand from a
  browser, but they carry the outcome of every dialogue. The excluded third was mostly where the
  no-deals were: 52.9% of it ends in a deal, against 85.8% of the rest.
- *Old and new results can be compared.* For the 4,463 dialogues of the Hugging Face copy, the
  new loader gives exactly the same turns and labels. The recovered dialogues are split on their
  own, so the 4,462 used before keep their split and the annotation sample stays in test.
- *Button presses are removed.* Offer, accept, reject and quit are structured actions, not typed
  messages. They are used to derive the outcome and then dropped. Anything typed after an accept
  or quit is dropped too. Text typed after a *rejected* offer is kept, because bargaining
  continues there (this happens in 1,491 dialogues).
- *Dialogues that just stop* (634 with no accept, reject or quit) count as no deal.
- *Very short dialogues are dropped.* 351 dialogues have fewer than two messages before the
  outcome (133 have none); they are removed, which leaves 6,331.
- *It is imbalanced.* 77.3% end in a deal, leaving 1,435 negatives, of which 216 are in the test
  split.

**Persuasion for Good**
- The label is the **actual** donation recorded after the chat, not what the person said they
  would give. The two differ: some people say "I'll donate" and then give nothing.
- The file with donations also holds personality surveys and demographics. None of it is used.

**SaaS Sales Conversations**
- The dataset card states it is entirely synthetic. It is labelled as synthetic everywhere,
  including in the dashboard, and is used for development and sanity checks, not as evidence.
- It ships with columns that give the answer away: a per-turn "conversion probability", and two
  scores that correlate 0.78 and 0.86 with the outcome. We read only the message text and the
  label.
- It also ships 3,072 embedding columns (about 3 GB), which we did not download.

### What counts as leakage and what does not

Removed: system-generated action turns, final prices, donation amounts from the survey file,
generated annotation columns, time-skip markers.

Kept: anything a participant typed. If the buyer writes "ok, $7, deal" before pressing accept,
that sentence stays. A real sales chat would contain it too, and a live system would see it.
The consequence is that scores near the end of a conversation are easy to get right, so we report
how early the score becomes accurate, not only how accurate it is at the end.

## 5. System overview

```
raw datasets ─► loaders ─► standard tables ─► cue tagging ─► scoring models ─► calibration
                (leakage     (turns,           (rules,        (v1, v2, v3)         │
                 removed,     conversations,    optional LLM)                      ▼
                 splits)      splits)                                   turning points
                                                                        explanations
                                                                        script insights
                                                                               │
                                                     evaluation ◄── exported files ──► dashboard
```

The dashboard only reads precomputed files. It never trains a model or calls an LLM.

| stage | state |
|---|---|
| Download, inspection, loaders, leakage removal, splits, tests | **done** |
| Rule-based cue tagger, v1 model, calibration, turning points (method A), template explanations, insights, export, basic evaluation | **done** |
| Dashboard: Home, Lead List, Conversation View, Script Insights | **done** (on v1) |
| Annotation tool and guidelines | **done** |
| Labels on the 60 conversations | one **LLM** annotator done; human labels **still to do, by us** |
| LLM cue tagger, LLM rubric scorer (v2), LLM rationales | **code done and tested; full run paused** (about a quarter of the cue tagging finished, v2 scoring and rationales not started). Needs a Gemini API key, about 3.5 hours and about $3 |
| Fine-tuned transformer (v3) | **done**, with `distilroberta-base` instead of ModernBERT-base |
| Full evaluation and report (change points, faithfulness, cross-domain, calibration, label agreement, case studies) | **done** |
| Final polish: dashboard checked in real screenshots and fixed, README rewritten as a from-scratch guide | **done** |

**What is left:** human labels on the 60 conversations (section 8a), finishing the Gemini run
(`make llm`, then `make export-v2` and `make eval`), and optionally retraining the SaaS
transformer on all of its training data.

## 6. Methods

Built and run: the rule-based tagger, the v1 and v3 models, calibration, both turning-point
methods, template explanations with the faithfulness check, and script insights. Built but not
yet run in full: the LLM tagger, v2 and LLM-written rationales.

### Cues
A fixed taxonomy of things a customer can signal and things an agent can do.

- **Customer cues, positive:** asks about features, asks about the buying process, asks the price
  neutrally, positive sentiment, commitment, agreement.
- **Customer cues, negative:** price objection, mentions a competitor or alternative, delay,
  doubt or distrust, disengagement, explicit refusal.
- **Agent moves:** gives a discount, states the price, explains a benefit, asks a question,
  builds rapport, addresses an objection, creates urgency, provides evidence, pushes to close,
  mentions a cost or fee.

Two taggers produce the same output format: a rule-based one (keywords and regular expressions,
always available) and an LLM one (Gemini, on a sample, optional).

### Three scoring models

| | model | role |
|---|---|---|
| v1 | Logistic regression on cue counts (cumulative and last 3 turns), agent-move counts, turn index, customer message length and TF-IDF of the customer's text | Interpretable baseline. Its coefficients drive the explanations. |
| v2 | An LLM given the transcript so far and a rubric, returning a 0–100 score | Optional comparison, on a small test sample. |
| v3 | Fine-tuned transformer on `[CUSTOMER] … [AGENT] …` text, keeping the most recent turns when the input is too long. Planned: ModernBERT-base. Built: `distilroberta-base`, 512 tokens, because of GPU memory. Class-weighted loss, up to 8 prefixes per conversation, early stopping on validation AUC. | Intended main model. Wins on one dataset of three so far. |

All scores are calibrated on the validation split (isotonic regression, or Platt scaling when the
validation set is small) so that a score of 0.7 means roughly a 70% chance.

### Turning points
A turn is a turning point when the score changes by at least 0.15 from the previous turn. A
second method, change-point detection (PELT), is compared against it.

### Explanations
For each turning point: the message where it happened, the agent message just before it, the cues
on it, the model features that contributed most, and a one-sentence rationale. Each explanation
gets a **faithfulness check**: we delete the trigger message, re-score, and see how much the
score moves. If it barely moves, the explanation is marked weak.

### Script insights
For each agent move, the average change in the customer's score on the customer's next turn, with
a bootstrap confidence interval, split by early, middle and late in the conversation. These are
associations, not proven causes, and the dashboard says so.

## 7. The dashboard

Built with Streamlit and working for all three datasets. Start it with `.\make.ps1 app`. A
sidebar chooses the dataset, model and split, and the synthetic dataset carries a badge on every
page.

**Suggested demo path (about three minutes):** open the Lead List on Persuasion for Good, turn
on evaluation mode to show how many of the top 20 really donated, open a conversation with a
turning point, click the triangle to show the explanation card, drag the replay slider back to
show the live view, then finish on Script Insights for the SaaS dataset.

1. **Lead List:** conversations ranked by score, with a status (hot, rising, cooling, stalled).
2. **Conversation View:** the transcript next to its score curve, clickable turning points with
   explanations, and a replay slider that shows the conversation as it would have unfolded live.
3. **Script Insights:** which agent moves are followed by rises or drops.

True outcomes are hidden unless an "evaluation mode" switch is turned on.

## 8. Evaluation plan

All on the test split, per dataset and per model.

| question | measure |
|---|---|
| Does the final score predict the outcome? | ROC AUC, PR AUC, Brier score, against the base rate |
| How early does it work? | AUC using the score at 10%, 25%, 50%, 75% and 100% of the conversation. **Key figure.** |
| Is the ranking useful? | Share of the top 10 and top 20 leads that converted, against the base rate |
| Are the probabilities honest? | Reliability diagram, expected calibration error |
| Do the score's movements match human judgement? | Agreement with our own hand labels on 60 conversations; Cohen's kappa between annotators; precision and recall of turning points |
| Are the explanations faithful? | Score change when the trigger turn is removed, versus a random turn |
| Does it transfer across domains? | Train on one dataset, test on the others |

### Results so far: v1 baseline and v3 transformer

Test split, final-turn score. Full tables are in `eval/report.md`. Test sets hold 951
(Craigslist), 153 (Persuasion for Good) and 3,000 (SaaS) conversations.

| dataset | base rate | model | ROC AUC (95% CI) | PR AUC | Brier (base-rate baseline) | top 10 converted | top 20 converted |
|---|---|---|---|---|---|---|---|
| CraigslistBargains | 77.3% | v1 | 0.827 (0.794–0.859) | 0.930 | 0.152 (0.176) | 100% | 100% |
| | | v3 | **0.908** (0.887–0.929) | 0.965 | 0.129 (0.176) | 100% | 100% |
| Persuasion for Good | 53.6% | v1 | 0.616 (0.527–0.701) | 0.636 | 0.247 (0.249) | 70% | 70% |
| | | v3 | 0.609 (0.514–0.693) | 0.672 | 0.252 (0.249) | 80% | 80% |
| SaaS (synthetic) | 49.9% | v1 | **0.778** (0.762–0.793) | 0.760 | 0.202 (0.250) | 100% | 95% |
| | | v3 | 0.599 (0.580–0.617) | 0.572 | 0.243 (0.250) | 80% | 80% |

**What the full Craigslist release changed.** The first version used the Hugging Face copy
(4,462 dialogues, 670 in test, 85.8% deals). With the full release:

| | first version | full release |
|---|---|---|
| Craigslist test conversations (no-deals) | 670 (95) | 951 (216) |
| v1 test ROC AUC | 0.746 | 0.827 |
| v3 test ROC AUC | 0.880 | 0.908 |
| v1 / v3 ROC AUC at 50% of the conversation | 0.585 / 0.629 | 0.732 / 0.741 |
| v1 on the 670 test conversations both versions share | 0.746 | 0.768 |

The gain on the shared conversations is smaller than the overall one, so part of the overall
gain comes from the recovered conversations being easier to predict. The intervals are now
narrower because the test split has more than twice as many no-deals.

**Early prediction (ROC AUC by share of the conversation seen):**

| dataset | model | 10% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|
| CraigslistBargains | v1 | 0.488 | 0.614 | 0.732 | 0.789 | 0.827 |
| | v3 | 0.570 | 0.600 | 0.741 | 0.821 | 0.908 |
| Persuasion for Good | v1 | 0.586 | 0.550 | 0.605 | 0.609 | 0.616 |
| | v3 | 0.526 | 0.507 | 0.501 | 0.547 | 0.609 |
| SaaS (synthetic) | v1 | 0.596 | 0.605 | 0.613 | 0.670 | 0.778 |
| | v3 | 0.545 | 0.557 | 0.571 | 0.589 | 0.599 |

Figures: `outputs/figures/early_prediction_v1.png` and `early_prediction_v3.png`.

**Important: v3 is not the model we planned.** The plan was ModernBERT-base. It did not fit in
the 4 GB GPU used for the first version (a GTX 1650 Ti) at the input lengths these conversations
need, so we used the plan's stated fallback, `distilroberta-base`, a smaller model. Inputs are
capped at 512 tokens, keeping the most recent turns. For SaaS it was trained on 4,000 of the
14,000 training conversations to keep the first run to about 2.3 hours in total. v3 scores exist
for the validation and test splits only. For the full Craigslist release, v3 was retrained for
all three datasets on an 8 GB RTX 4070 Laptop GPU, in about 1.4 hours, with the same model and
settings so the two versions can be compared.

**How to read these.**
- **Craigslist: the transformer is clearly better** (0.91 against 0.83; the intervals do not
  overlap). Reading the whole negotiation helps more than counting cues, and the gain shows from
  the three-quarter point on.
- **Persuasion for Good: no difference.** Both models are near 0.61 with wide, overlapping
  intervals, and neither clearly beats the base-rate Brier score. With 711 training conversations and a
  noisy label (people say they will donate and then do not), a bigger model has nothing more to
  learn from. This looks like a limit of the data, not of the model.
- **SaaS: the transformer is worse than the baseline** (0.60 against 0.78). Two likely reasons,
  neither tested yet: it saw under a third of the training conversations that v1 saw, and its
  validation score stayed low (about 0.55), so it stopped early. The first version's run gave
  0.667 on the same, unchanged SaaS data, so v3 on SaaS also varies a lot from run to run. We
  should say this plainly and not present v3 as the better model overall.
- **Early prediction is still weak outside Craigslist.** At the halfway point Craigslist reaches
  0.73 (v1) and 0.74 (v3); the other datasets are between 0.50 and 0.61. Most of the signal
  still arrives in the last quarter of the conversation.
- Craigslist's 100% in the top 20 is less impressive than it looks, because 77% convert anyway.
- v1 overfits (train AUC 0.84 to 0.91 against 0.62 to 0.83 on test). Neither model has been tuned.

**Turning points and explanations (threshold 0.15).** v1 is counted over all splits; v3 only
over the validation and test splits, which is all it scored.

| dataset | model | turns scored | turning points | flagged weak |
|---|---|---|---|---|
| CraigslistBargains | v1 | 49,091 | 1,858 | 3% |
| | v3 | 14,653 | 913 | 13% |
| Persuasion for Good | v1 | 20,932 | 662 | 1% |
| | v3 | 6,315 | 347 | 24% |
| SaaS (synthetic) | v1 | 243,197 | 5,590 | under 1% |
| | v3 | 72,826 | 94 | 21% |

- For v1, removing the trigger message changes the score by 0.17 to 0.19 on average, so the
  turns it points to do drive its score.
- For v3, more explanations are weak (13% to 24%). The transformer reads the whole context, so
  removing one message more often leaves the score where it was. v3 also gives no per-feature
  breakdown; its explanation is the cue-based sentence plus the faithfulness check.
- v3 finds more turning points per conversation than v1 on Craigslist, and far fewer on SaaS,
  where its scores are flatter.

### Full evaluation findings (Phase 7)

All on the test split. Full tables, figures and case studies: `eval/report.md`.

**1. Explanations are faithful.** Removing the message an explanation points to moves the score
far more than removing a random other message from the same conversation.

| dataset | model | turning points | trigger removed | random turn removed | ratio |
|---|---|---|---|---|---|
| CraigslistBargains | v1 | 263 | 0.172 | 0.026 | 6.6x |
| | v3 | 482 | 0.136 | 0.049 | 2.8x |
| Persuasion for Good | v1 | 105 | 0.191 | 0.031 | 6.2x |
| | v3 | 180 | 0.174 | 0.043 | 4.1x |
| SaaS (synthetic) | v1 | 855 | 0.183 | 0.032 | 5.7x |
| | v3 | 47 | 0.148 | 0.043 | 3.4x |

This is the strongest positive result about the explanations. One caveat: a turning point is
defined by a jump at that message, so some effect from removing it is expected by construction.
The random-turn baseline shows it is specific to that message.

**2. The score's turn-by-turn movements do not match the annotator.** This is the weakest
result, and it should be reported as such. Against the LLM annotator (there are no human labels):

- *Direction (up / same / down) on 382 customer turns:* v1 is right 45.5% of the time and v3
  48.2%. Always answering "same" would be right 51.8% of the time. Cohen's kappa is 0.17 for v1
  and 0.08 for v3, which is little agreement beyond chance. The one exception is v1 on
  Craigslist: 64.3% against 37.1% for always "same" (kappa 0.34), but on only 70 turns.
- *Turning points (within one turn):* precision is 12% to 21% and recall 12% to 24% across both
  models and both detection methods. The annotator marked 33 turning points; the models marked
  32 to 52, mostly in different places.

So the score ranks conversations reasonably by how they end, but the individual ups and downs
of the curve are not the ones a reader would pick. Possible reasons: the models are trained only
on final outcomes, never on turn-level judgements; the annotator is an LLM and one annotator;
and the direction result depends on the ±0.02 band used to call a change "same".

**3. The two turning-point methods disagree with the annotator about equally.** Change-point
detection (method B) finds more points than the score-jump rule (method A) on most datasets, for
example 1.05 against 0.69 per conversation on Persuasion for Good with v1. Neither is clearly
better against the labels (F1 0.12 to 0.19 overall).

**4. Models do not transfer across domains.** ROC AUC when trained on the row and tested on the
column (in-domain on the diagonal):

| v3 trained on | CraigslistBargains | Persuasion for Good | SaaS |
|---|---|---|---|
| CraigslistBargains | **0.909** | 0.619 | 0.633 |
| Persuasion for Good | 0.618 | **0.609** | 0.490 |
| SaaS (synthetic) | 0.542 | 0.507 | **0.603** |

v1 shows the same pattern (off-diagonal 0.52 to 0.60). Two things stand out. The Craigslist
transformer scores 0.619 on Persuasion for Good, as good as the model trained on that data,
which again suggests that dataset has a low ceiling. And the transformer trained on synthetic
SaaS data is close to chance on both real datasets (0.54 and 0.51), so what it learned from the
synthetic conversations does not carry over to real ones.

**5. Calibration is acceptable but worse on Craigslist than before.** Expected calibration
error is 0.03 to 0.11, and 0.13 for v3 on Persuasion for Good. On Craigslist it rose with the
full release (v1: 0.094, against 0.036 in the first version): scores in the middle range are too
high, which is where many of the recovered no-deal conversations sit. Figures: `reliability_v1.png`,
`reliability_v3.png`.

**6. Case studies.** `eval/report.md` section 8 lists, per dataset, three confident correct
predictions, three confident mistakes and the three largest turning points, with conversation
IDs that open directly in the dashboard. They are ready to use in the presentation.

**Script insights (test split), largest effects only:**
- *SaaS:* after the agent states a price, the customer's score falls on their next turn
  (−0.020 on average, a drop 58% of the time); mentioning a cost or fee is similar (−0.017).
- *Persuasion for Good:* every agent move is followed by a small average rise (about +0.01 to
  +0.02). The effects are small and the intervals for the rarer moves are wide.
- *Craigslist:* every agent move is followed by a rise, larger with the full release (about
  +0.05 to +0.08 on average, largest for asking a question and building rapport early on).
  Almost all v1 turning points on Craigslist are rises too (97%), so this mostly reflects the
  score drifting upward as a negotiation goes on, not an effect of particular moves.

These figures are for v1. They are associations measured with a weak baseline; we would not
present them as findings yet.

## 8a. Hand labelling: what the group needs to do

The outcome label only tells us how a conversation ended. To check whether the score *moves*
sensibly in the middle, and whether its turning points are real, we need human judgement.

- **What:** 60 fixed conversations, 20 per dataset, all from the test split
  (`annotation/sample.json`). Everyone labels the same 60.
- **How:** `.\make.ps1 app`, open **Annotate**, type your name. For each customer turn choose
  *more / same / less inclined than before*, and tick *turning point* on any turn where the
  conversation clearly changed course. Model scores, cues and outcomes are hidden on that page.
- **Rules:** work alone, read top to bottom, do not look the conversation up in the other views
  first. Full instructions with three worked examples: `annotation/guidelines.md`.
- **Effort:** about 800 turns in total, roughly 400 of them customer turns needing a choice.
- **Output:** `annotation/labels/<name>.jsonl`, one file per person. Send the file to whoever
  runs the evaluation.
- **At least two annotators are needed** so we can report agreement between people (Cohen's
  kappa). Without that we cannot say how reliable the labels themselves are.

Phase 7 then reports how often the score's direction matches the human label, and the precision
and recall of the model's turning points against the human ones.

**Current state of the labels (2026-10-08): one annotator, and it is an LLM.** No person has
labelled yet. To have a reference in the meantime, the 60 conversations were labelled by an LLM,
following the same guidelines and reading only the transcripts, without outcomes, model scores
or cue tags. The file is `annotation/labels/auto_llm.jsonl`; the readable source is
`annotation/llm_labels.txt`.

- 416 customer turns: 138 more inclined, 207 same, 71 less inclined. 33 turning points.
- **These must be reported as LLM annotations, not human ones.** Any agreement number computed
  against them is "agreement with an LLM annotator".
- They do not give us inter-annotator agreement. If one or two of us label the same 60, we get
  human labels, human-to-human kappa, and human-to-LLM kappa, which is a stronger result.
- An LLM annotator may share blind spots with the models being evaluated (for example, leaning
  on the same surface cues), so agreement with it is weaker evidence than agreement with people.

## 9. Limitations to state up front

- **Proxy data.** Bargaining and charity donation stand in for sales. The findings may not carry
  over to real sales chats.
- **Synthetic data.** The SaaS dataset is LLM-generated, and many of its conversations end
  without a visible decision. Results on it show the pipeline works, not that the method works.
- **Small and imbalanced test sets.** Persuasion for Good has 153 test conversations, so its
  confidence intervals are wide. Craigslist has 216 negatives in test (95 in the first version).
- **Noisy labels.** In Persuasion for Good, stated intent and actual donation differ. In all
  datasets, an early prefix carries the label of an ending that was not yet decided.
- **Late-turn scores are easy.** People often state their decision in words near the end.
- **The cue tagger is keyword-based.** It misses paraphrases and misfires on some words; several
  cues are rare (refusal, delay and competitor mentions each tag under 1% of Craigslist turns).
- **Fixed thresholds do not fit every dataset.** Even with the full Craigslist release (77% base
  rate), about 96% of Craigslist test leads are labelled "Hot" under v1.
- **v1 explanations lean on single words.** The top contributing features are often words such
  as "okay" or "demo", which are less readable than cues.
- **Insights are correlational.**

## 10. Questions we should expect

**Isn't labelling every prefix with the final outcome wrong?** It is noisy, not wrong. The target
is the probability of the final outcome given what has been seen so far, and training a
classifier on prefixes with the final label estimates exactly that. Early prefixes should, and
do, get scores near the base rate.

**How do you know the model is not cheating?** Outcome-revealing turns and fields are removed in
the loaders. Automated tests check that the turn table has only five columns (dataset,
conversation id, turn, role, text), that no action or system turns survive, and that splits never
separate one conversation. Another test changes the turns after turn *k* and confirms the scores
up to *k* stay identical, for each trained model. By construction, every v1 feature is a sum over
the turns seen so far.

**Why not use the Craigslist dataset's own train/test split?** Its test split had no labels in
the Hugging Face copy used first, and keeping our own split lets old and new results be compared
on the same test conversations.

**Why keep the synthetic dataset at all?** It is the only one that looks like sales, it is large
and balanced, and it is useful for developing and debugging. It is never presented as real.

**Why logistic regression when a transformer is available?** It gives a baseline to beat, runs
without a GPU, and its coefficients make the explanations direct.

**How is a turning point different from noise?** That is what the hand labels and the
faithfulness check are for. We do not claim it yet.

## 11. Reproducing it

Save `train.json`, `dev.json` and `test.json` from the CodaLab worksheet into
`data/raw/craigslist/` first (see `data/README.md`); the other datasets download automatically.

```
.\make.ps1 setup      # virtualenv and packages
.\make.ps1 data       # download and build the standard tables
.\make.ps1 train      # train the v1 model per dataset (about 2 minutes)
.\make.ps1 export     # scores, cues, turning points, leads, insights into outputs/
.\make.ps1 eval       # metrics.json, figures, eval/report.md
.\make.ps1 train-v3   # optional: fine-tune the transformer (about 2.3 hours on a 4 GB GPU, 1.4 on an 8 GB RTX 4070)
.\make.ps1 test       # 81 tests: loaders, leakage, splits, scoring, outputs, evaluation, dashboard, annotation, LLM jobs
.\make.ps1 all        # data, train, export and eval in one go
.\make.ps1 app        # open the dashboard
```

(`make <target>` on systems with GNU make.) Details: `README.md`, `data/README.md`.

## 12. Glossary

- **Turn:** one message from one speaker.
- **Prefix:** the first *k*+1 turns of a conversation.
- **Leakage:** information in the input that reveals the label.
- **Calibration:** adjusting scores so they behave like true probabilities.
- **ROC AUC:** the chance that a random buyer is scored above a random non-buyer; 0.5 is chance.
- **Brier score:** mean squared error of the predicted probability; lower is better.
- **Turning point:** a turn where the score changes sharply.
- **Faithfulness:** whether removing the thing an explanation points to really changes the score.
- **Cue:** a labelled signal in a message, such as a price objection.
