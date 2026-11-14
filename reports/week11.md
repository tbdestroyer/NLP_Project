# Weekly Project Report — Week 11

**Team Name:**  
**Date:** 

## Team Members

| Name | Email | GitHub Username | Role |
|---|---|---|---|
| Taner Bulbul | tbulbul@umd.edu | tbdestroyer | Product Lead / Developer |
| Ayesha Khan | ayekhan@umd.edu | ayekhan12 | Product Idea / DS |
| Namratha Jeetendra | namrath4@umd.edu | namrathajeetendra | |
| Shantanu Ramavat | sramavat@umd.edu | sramavat-24 | Software Engineer |
| Vaibhav Devarapalli | gdevarap@umd.edu | | |

## 1. Shipped This Week

- Ran one moderated usability session covering five app tasks. The
  [cleaned transcript](../user_evidence/2026-11-14_user1/transcript.txt) is in `user_evidence/`.


## 2. User / Validation Learning

- The participant could use high lead percentages to identify likely buyers, but said nearly
  every lead showed about 99%, making the scores less useful for prioritization.
- In Conversation View, the participant noticed a buyer's price objection but was unsure how to
  find the point where interest dropped.
- In Analyze, the participant was unsure how to interpret the final score and its apparent
  conflict with the negative analysis. They did see value in analyzing another captured
  conversation and its turning points.
- Insights' empty “What hurts” section was confusing without context about what the results
  represent. The participant also did not understand how Live would fit into an actual sales call
  or how the demo call was selected.
- The participant saw potential value for sales and marketing teams reviewing conversations to
  improve customer interactions. Treat these as observations from one session, not validated
  general conclusions.


## 3. Metrics Snapshot

- Qualitative usability sessions: 1; tasks attempted: 5.
- The transcript does not record task timings or establish a completion or success rate.
- The participant reported slow page loading and difficulty scrolling.


## 4. Challenges / Blockers

- Clarify score meaning and direction, especially how the final score relates to the analysis.
- Make the empty “What hurts” state and the scope of Insights understandable.
- Explain the Live workflow and distinguish a demo call from a user's own live conversation.
- Investigate the reported slowness and scrolling difficulty.


## 5. Next Week's Goals

- Review score labels and explanations so users can interpret likelihood and turn-by-turn changes.
- Add clearer context for Insights, including when there are no “What hurts” results.
- Make the Live entry point, demo behavior, and intended real-call workflow explicit.
- Reproduce and measure the reported page-load and scrolling delays.
- Run additional usability sessions before treating these observations as representative.


## 6. Individual Contributions

### Taner Bulbul

**Contributions:** Conducted the user-testing session, prepared and archived the cleaned
transcript, and documented the findings, usability challenges, and follow-up goals in the
Week 11 report.

**GitHub Evidence:** [Week 11 report](week11.md) and
[cleaned transcript](../user_evidence/2026-11-14_user1/transcript.txt).

### Ayesha Khan

**Contributions:** Set up the full pipeline on my Mac and added Apple-GPU (mps) support so the v3 transformer trains and scores there. Brought in the v3 outputs from the team's GPU run and reran the evaluation, so the report now includes results against Taner's human labels (human vs. LLM annotator agreement: kappa 0.42 on direction, 0.10 on turning points). Fixed precision@10/20, which changed between runs because many leads share the same calibrated score. Updated PROJECT_EXPLAINER.md, both READMEs and the presentation slides to match, and opened PR #6 to merge the changes into main.

**GitHub Evidence:** PR [#6](https://github.com/tbdestroyer/NLP_Project/pull/6): `ecef2c3` (v3 on Apple GPUs), `ceec7f9` (evaluation rerun with Taner's labels), `734e5eb` (precision@k fix), `8546b0e` (docs update)

### Namratha Jeetendra

**Contributions:**

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

No canvas changes made based on this single usability session; follow up with additional users.
