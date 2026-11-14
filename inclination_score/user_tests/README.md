# User test uploads

Drop user-test materials into the matching folder:

- `test_cases/` — test scenarios, tasks, questionnaires, or notes about what to observe.
- `transcripts/` — transcripts and notes from sessions that ran those tests.

Use the same short ID for related files, for example `session-001.md` in each folder.
These are research materials, not automated pytest tests; the automated test suite lives in
`tests/`.

Files added to these folders are ignored by Git by default, since transcripts may contain
personal or otherwise sensitive information. Remove identifying details before sharing or
committing any materials. Materials intended to be shared in Git can be placed in the
repository's `user_evidence/` folder after review.
