# Session notes — user2

Reconstructed on 2026-11-27 from the screen recording and facilitator-supplied transcript, not contemporaneous notes. The original screen recording was captured during the test on 2026-11-26.

- **Date:** 2026-11-26
- **Participant:** user2, external participant; project manager, no sales experience. Age/gender omitted from repository evidence as unnecessary identifiers.
- **Product version / commit:** running current project build per facilitator; exact Git SHA not recorded. UI shows CraigslistBargains / v1 / Test for the initial workflow; Analyze displays the SaaS synthetic model for sales chats.
- **Method:** think-aloud tasks on running website; no coaching intended. Facilitator repeated/rephrased T3 and discussed the protocol before moving on. Screen recording duration: 6:24 (384 s).
- **Raw artifact:** [Screen recording](https://drive.google.com/file/d/14Igz_DVYt6A8cyPsMPCM-qbRCbzPUzuk/view?usp=sharing) (Drive sharing metadata confirms file name; viewer access not verified. Review raw video/audio for personal information and participant consent before sharing.)
- **Edited transcript:** [transcript.txt](transcript.txt); [task log](task_log.csv). Screenshots are extracted from the raw recording, cropped to exclude browser chrome and desktop.

## Observations

| Task | Observation | Assessment |
|---|---|---|
| T1: Prioritize leads | Chose leads based on Hot/99%; did not understand what the percentage meant. At 1:00 the board shows 906 Hot out of 951 leads. | Partial; no individual rationale beyond status/score. |
| T2: Find the drop | Reported slow clicks; pointed at a turn in Crestline tandem bike. At 1:40 the transcript shows buyer turn 3, 63%, −12 points, no threshold-defined turning point. | Partial; specific selection and correct curve identification not independently established. |
| T3: Score a new chat | Asked where to go, what to read and where to enter the chat. Analyze form visible at 3:30; no scoring result established. Participant said they did not know and moved on. | Not completed; repeated task prompting, no valid score interpretation. |
| T4: Identify helpful/harmful moves | Identified Builds rapport; inferred Gives discount was unhelpful; complained graph lacked figures. At 4:15 What hurts was empty and a tooltip showed +5.8 points for a move. | Partial; no actual harmful move identified. |
| T5: Live demo extension | Described interest cooling and improving after a deal. At 6:15 the Live transcript and coaching panel are visible. | Qualitative interpretation; no predefined success criterion, real call or coaching decision verified. |

## Recording anchors (not task timings)

- [T1 — 01:00](screenshot_T1_01m00s.png): lead board, high scores.
- [T2 — 01:40](screenshot_T2_01m40s.png): buyer turn 3 and score drop.
- [T3 — 03:30](screenshot_T3_03m30s.png): Analyze form, chat pasted, no result visible.
- [T4 — 04:15](screenshot_T4_04m15s.png): Insights, tooltip, empty harmful-moves panel.
- [T5 — 06:15](screenshot_T5_06m15s.png): Live transcript and suggestions.

These are elapsed video positions, not measured task start/end times. Formal completion and seconds fields are left blank to match the first session's conservative reporting; narrative assessments are in the log. No success rate is claimed.

## Findings and proposed changes (not verified as shipped)

1. Both users relied on high percentages and had difficulty interpreting scores. Explain probability and dataset-specific score/threshold behavior; review lead prioritization defaults.
2. Both struggled with Analyze. Provide an obvious paste/upload → score → interpret sequence and loading/error feedback; test an unassisted new-chat task.
3. Both encountered an empty What hurts panel; user2 additionally struggled to interpret chart values. Explain absence of qualifying negative moves and make numeric labels and uncertainty more visible.
4. Both reported slow interactions; investigate client/API latency and add visible progress/loading states.
5. User2 could describe a Live conversation's change in interest, but neither test establishes that the coaching is useful in a real sales call. Test with a sales practitioner.

Limitations: two external sessions so far, no verified sales experience for user2, no formal task times, unknown tested commit, and no verified post-fix retest. These artifacts are prepared for commit; repository commit date and any fixes must be verified separately.
