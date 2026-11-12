# Weekly Project Report — Week 10

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



## 2. User / Validation Learning



## 3. Metrics Snapshot



## 4. Challenges / Blockers



## 5. Next Week's Goals



## 6. Individual Contributions

### Taner Bulbul

**Contributions:**
Last section that we need to do is select roles for presentations which we will do Wednesday during our meeting, and do the user evidence. I have sent emails and text messages to friends and neighbors asking if any of them would be willing to be a user and test the website. 

**GitHub Evidence:**
No commit evidence for this contribution, evidence includes PII. 

### Ayesha Khan

**Contributions:** Created draft presentation with the outline for what each of the 5 sections needs to cover for our product specifically. Discussed it with the team in the project meeting. Communicated that the next open item for us to cover is user evidence. Also liaising to assign each section to a team member as the agenda for our next week project call.

**GitHub Evidence:**

### Namratha Jeetendra

**Contributions:**

**GitHub Evidence:**

### Shantanu Ramavat

**Contributions:** Built Live mode (`#/live`), a real-time conversation tracker that captures microphone and shared-tab call audio through an AudioWorklet pipeline (16 kHz resampling, voice activity detection and echo suppression) and transcribes each utterance with a locally hosted Whisper `small.en` model on the GPU (about 0.9 s per sentence). Each utterance is sent to a new `/api/live/score` endpoint, which re-scores the transcript so far with the existing inclination model and updates the customer's purchase probability, score curve, cues and lead status (Hot, Rising, Cooling or Stalled) after every message. I also added a data-driven coaching engine (no LLM) that builds a playbook from the held-out validation and test conversations, measuring how the customer's next-reply score shifted after each agent move given the customer's preceding cues. During a call it ranks candidate moves against the typical reply in the current situation, using shrinkage for rarely seen moves and 95% intervals to label them Strong evidence, Suggestive, About typical or Avoid. The page also supports mic-only, typed and spoken-demo input (Web Speech API voices with word-by-word captions), and the backend is covered by 18 new unit tests.

**GitHub Evidence:** Commit `19ddb21` on `shantanu/dev`, merged into `main`: `inclination_score/src/inclination/live.py`, `inclination_score/api/server.py`, `inclination_score/tests/test_live.py`, `web/src/pages/Live.jsx`, `web/src/live/audio.js`, `web/src/live/voices.js`.

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
