# User evidence

This folder holds raw evidence that real users used the running product (the sales workspace in
`web/`, backed by the API in `inclination_score/api/`). Evidence only counts if it is captured
during the session and committed in the same week as the claim it supports.

## What counts

- A screen or session recording of someone using the product
- A timestamped usage log of a real user's session
- A task test: a task given to a user, with whether they completed it and how long it took
- A dated screenshot of the product in the user's hands, with notes taken during the session

## What does not count

- Interviews about the idea, or reactions to mockups, slides or descriptions
- Feedback on the report instead of the product
- Sessions run by our own team members
- Summaries written later from memory

## Protocol

1. **Recruit** 3-5 participants outside the team (ideally with sales or support experience).
   Assign an anonymous ID (`user1`, `user2`, ...). Never commit names, emails or faces.
2. **Prepare** the product: start the API (`.\make.ps1 api` in `inclination_score/`) and the UI
   (`npm run dev` in `web/`). Record the commit or version tested in `notes.md`.
3. **Run the session** on the running product. Read the task aloud, do not demo or coach, and
   start the recording and timer. Take notes during the session.
4. **Capture** a recording, the task log row(s), and a dated screenshot with notes.
5. **Commit the same week** in a folder `user_evidence/YYYY-MM-DD_userN/`.
6. **Anonymize** before committing: blur faces, emails and any personal data in recordings,
   screenshots, logs and uploaded chats. Large videos may be stored elsewhere, but commit a
   timestamped log or screenshot plus a link in `notes.md`.

## Tasks

| ID | Task | Success criterion |
|---|---|---|
| T1 | Find the three hottest leads and say why each is hot | Names three leads and cites at least one reason each |
| T2 | Open a conversation and identify the turning point where interest dropped | Points to the correct turn on the score curve |
| T3 | Upload a chat in Analyze and say whether the lead looks likely to buy | Reads the score and status correctly |
| T4 | Say which sales moves help or hurt most | Identifies a helpful and a harmful move on Insights |

Adjust the tasks as the product changes and note the change in `notes.md`.

## Session folder contents

```
user_evidence/YYYY-MM-DD_userN/
  notes.md          copy of session_notes_template.md, filled in during the session
  task_log.csv      copy of task_log_template.csv, filled in during the session
  screenshot_*.png  dated screenshots, anonymized
  recording link or file, anonymized
```

## Sessions

| Date | Participant | Folder | Tasks completed |
|---|---|---|---|
| 2026-11-14 | user1 | [Session folder](2026-11-14_user1/) | Not established; see notes |
| 2026-11-26 | user2 | [Session folder](2026-11-26_user2/) | T1, T2 and T4 partial; T3 not completed; T5 qualitative only |

### user2 session

An external participant tested the running web app using think-aloud tasks. The [notes](2026-11-26_user2/notes.md) and [task log](2026-11-26_user2/task_log.csv)
were reconstructed from the recording and supplied transcript on 2026-11-27.

- **T1:** Selected high-scoring Hot leads but did not understand the percentages or give individual reasons.
- **T2:** Pointed to a score drop and reported slow clicks; correct turning-point identification was not established.
- **T3:** Struggled to find where to enter and score a chat; the task was not completed.
- **T4:** Identified rapport as helpful but did not identify a confirmed harmful move; the harmful-moves panel was empty and chart values were unclear.
- **T5:** Described interest cooling and then recovering in the Live demo. This was an additional qualitative task, not a test of real-call coaching.

The [session notes](2026-11-26_user2/notes.md) link the recording, transcript and screenshots.
Screenshot timestamps are recording positions. Review the recording for personal information and consent before sharing, and verify the evidence commit date separately.
