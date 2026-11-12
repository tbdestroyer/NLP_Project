# Live mode: score and coach a sales call while it happens

## What it does

The web app's **Live** page listens to a sales call and, after every message:

- writes the transcript (speech-to-text with Whisper, run locally on this computer)
- updates the customer's chance to buy, the score curve and the lead status (Hot, Rising, Cooling or Stalled)
- tells the salesperson **what to say next**: up to 3 suggested moves, each with a line to say,
  how strong the evidence is, and real lines from past conversations where that move worked
- lists the moves to **avoid right now**, and shows the latest turning point

Nothing is stored. Audio and transcripts stay in the browser and on the local data server.
**Download transcript** saves the call as a `.txt` file that the Analyze page can open.

## How to use it

```
cd inclination_score
make api        # terminal 1: data server (scoring + Whisper)

cd web
npm run dev     # terminal 2: open http://localhost:5173 and go to Live
```

1. Pick how you take the call:
   - **Microphone + call audio**: the microphone is you. For the customer, share the call's
     browser tab (Meet, Teams or Zoom on the web) and tick *Also share tab audio*. For a desktop
     app, share the entire screen with *Share system audio*. Wear headphones; if the microphone
     still picks up the customer, those near-copies are removed.
   - **Microphone only**: for a meeting in person or on speakerphone. Press **Space** whenever the
     other person starts talking.
   - **Type, or play a demo**: no audio. The demo plays a 14-line SaaS call with a price objection.
     With **Speak the demo aloud** on, the browser's built-in voices (Web Speech API) read it out,
     with a male voice for the sales rep and a female voice for the customer. Each line fills in
     word by word as it is spoken, and the full demo takes about 90 s. On Windows, Chrome uses
     voices like Microsoft David and Zira. Edge has more natural "Online" voices, which are chosen
     when present, so Edge is best for presenting. **Skip line**, **Pause demo** and
     **Voices on/off** work during the demo.
2. Pick the kind of conversation (sales, price negotiation, persuasion). v1 is selected: it
   updates in about 0.05 s. v3 takes a few seconds per update.
3. Click **Start listening**. Click a speaker's name to swap a message to the other side, or
   type a message to add or correct one.

Use Chrome or Edge on `http://localhost` (browsers allow the microphone only on localhost or HTTPS).

## How it works

```
microphone / shared tab
   │  AudioWorklet → resample to 16 kHz → voice activity detection (web/src/live/audio.js)
   ▼  one utterance (float32 samples)
POST /api/live/transcribe ──► Whisper small.en on the GPU (live.Transcriber) ──► text
   │
   ▼  transcript so far
POST /api/live/score ──► analyze.analyze()  (same scoring as the Analyze page)
                     └─► live.coach()       (the playbook below)
   ▼
score curve, status, cues, turning points, "what to say next"
```

**Speech-to-text.** `openai/whisper-small.en` through the `transformers` and `torch` packages
already in `requirements-ml.txt`, so no new packages are needed. It downloads once (about 1 GB, into `.cache/huggingface`)
and loads when the Live page opens, which takes about 25 s. On the GTX 1650 Ti an 8-second sentence
takes about 0.9 s and uses 1.1 GB of GPU memory. The model and thresholds are under `live:` in
`config.yaml`.

**The coaching playbook.** For every agent message in the held-out (val and test) conversations
of the chosen dataset, it records:

- the customer's previous cues (the *situation*, e.g. a price objection)
- the agent's move (e.g. providing evidence)
- how much the score changed at the customer's next reply

This is the same effect as Script insights, but split by situation. In a live call, the coach
reads the cues of the customer's latest message and ranks moves by how they compare with a
*typical reply in that situation*. Moves seen only a few times count for less (`live.shrinkage`).
Labels:

- **Strong evidence**: better than a typical reply, beyond the 95% interval
- **Suggestive**: better on average
- **About typical**: no worse than a typical reply
- **Avoid**: clearly worse than a typical reply

A situation with too little data falls back to all replies. The playbook builds in about 1 s
the first time it is needed, from the exported outputs. It is rebuilt when those files change.

These are associations in past conversations, not causes, and the page says so. The suggested
lines are templates with `[brackets]` to fill in; no LLM is used.

## Files

| File | What |
|---|---|
| `src/inclination/live.py` | `Transcriber`, `build_playbook`, `coach`, `live_update` |
| `api/server.py` | `GET /api/live/status`, `POST /api/live/load`, `/api/live/transcribe`, `/api/live/score` |
| `config.yaml` | `live:` settings |
| `tests/test_live.py` | 18 tests (Whisper is replaced by a fake) |
| `web/src/live/audio.js` | capture, resampling, voice activity detection, echo check |
| `web/src/live/voices.js` | demo voices: picking two voices, speaking a line, word-by-word captions |
| `web/src/pages/Live.jsx` | the Live page |

## Limits

- Coaching is only as good as the cue rules. For example, "more than we budgeted" is tagged as
  doubt, not as a price objection.
- The SaaS evidence comes from synthetic chats. Craigslist and Persuasion for Good are real
  conversations, but on different topics.
- Tab audio cannot be tested in headless Chrome; it was checked only with a fake microphone.
