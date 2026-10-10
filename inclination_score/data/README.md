# Data notes (Phase 0 inspection)

Produced by `make download` and `make inspect` on 2026-10-07. The full printed schemas and example
records are in `outputs/data_inspection.txt`. Everything below was read from the real files, not
from the spec.

## 1. CraigslistBargains (`craigslist`) — real humans

**Source used.** `stanfordnlp/craigslist_bargains` on Hugging Face, branch `refs/convert/parquet`
(Hugging Face's own parquet conversion). The loading script no longer runs with current
`datasets`, and the raw CoCoA JSON on CodaLab is unreachable (expired TLS certificate, and the
server answers HTTP 500 regardless).

**Structure.** 6,682 dialogues (HF splits: train 5,247 / validation 597 / test 838), one row each:

| field | content |
|---|---|
| `agent_info` | `Role` = `[buyer, seller]` (always index 0 = buyer), private `Target` price per role, `Bottomline` (always `"None"`) |
| `agent_turn` | list of speaker indices (0 = buyer, 1 = seller), one per turn |
| `utterance` | list of turn texts; **empty string for structured action turns** |
| `dialogue_acts` | `intent` and `price` per turn, from the CoCoA rule-based parser |
| `items` | `Category`, `Title`, `Description`, `Price` (listing price), `Images` |

Mean 9.2 turns per dialogue (min 2, max 46). Buyer speaks first in 57%. Six categories: furniture,
housing, bike, car, electronics, phone.

**Outcome field.** There is no outcome column. The outcome must be derived from
`dialogue_acts.intent`: `accept`, `reject`, `quit` (and `offer`) mark the structured action turns.

**Problem: a third of the dialogues have no dialogue acts.** 2,219 dialogues have `intent == ""`
on every turn: **all 838 of the HF test split**, 1,237 of train and 144 of validation. Their action
turns are still visible as empty utterances, but an `accept` cannot be told apart from a `reject`,
so their outcome is unknown. Only **4,463 dialogues are labelled**.

Among the 4,463 labelled dialogues:

| ends with | count | planned outcome |
|---|---|---|
| `accept` | 3,831 (85.8%) | 1 |
| `quit` | 216 | 0 |
| `reject` | 202 | 0 |
| an `offer` nobody answered | 20 | 0 |
| no action turn at all (abandoned) | 194 | 0 |

**Leakage risks.**
- Action turns (`offer`, `accept`, `reject`, `quit`): always an empty utterance, and the `offer`
  carries the final deal price. All are dropped from the turn table.
- Action turns are **not always at the end**: in 1,072 dialogues people keep typing after an
  offer (for example, an offer is rejected and bargaining continues). Those text turns are kept.
- `dialogue_acts.intent` / `.price` are annotations, not conversation content: never used as
  features.
- `agent_info.Target` (private target prices): display metadata only.
- Utterances contain no system-generated tokens such as `<accept>` in this version; the templated
  turns are simply empty.

## 2. Persuasion for Good (`p4g`) — real humans

**Source used.** Original release, `gitlab.com/ucdavisnlp/persuasionforgood`, `data/FullData/`.
(The ConvoKit download URL returns 404, so `convokit` is not needed and was not installed.)

**Structure.**
- `full_dialog.csv`, 20,932 rows, one per message: `Unit` (text), `Turn` (exchange index 0–14,
  shared by one persuader message and the reply), `B4` (role), `B2` (dialogue id).
- `full_info.csv`, 2,034 rows, one per participant per dialogue: `B2` (dialogue id), `B3` (user
  id), `B4` (role), `B6` (donation), `B7` (number of exchanges), then 23 personality / values
  scores and 8 demographic fields.

1,017 dialogues, mean 20.6 messages. `B4 == 0` is the persuader, `B4 == 1` the persuadee. The
persuader speaks first in every dialogue.

**Outcome field.** `B6` on the persuadee's row is the **actual** donation in dollars (median
$0.05, max $700). `B6 > 0` for **53.6%** of persuadees. There is no intended-donation column (`B5`)
in this file, so no fallback is needed.

**Leakage risks.**
- All of `full_info.csv` (donation, survey scores, demographics) stays out of the turn table.
  Only `B6` is read, and only to build the label.
- Donation amounts typed by the participants in the chat ("I would like to donate $0.20") are
  real conversation content and are kept. They make late turns very predictive, so the
  early-prediction curve is the more honest figure for this dataset.

## 3. SaaS Sales Conversations (`saas`) — **SYNTHETIC**

**Source used.** `DeepMostInnovations/saas-sales-conversations` on Hugging Face, parquet branch.

**Synthetic: yes.** The dataset card states it is "entirely synthetic, generated using Azure
OpenAI's GPT-4". It will be flagged `is_synthetic = True` and badged on every dashboard page.

**Structure.** 100,000 conversations, 3,088 columns, of which 3,072 are `embedding_*`. Only the 16
other columns were downloaded (197 MB instead of about 3 GB):

| field | content |
|---|---|
| `conversation_id` | unique |
| `conversation` | JSON list of `{speaker, message}`; speakers are `customer`, `sales_rep`, `system` |
| `outcome` | 0 / 1 |
| `company_name`, `product_name`, `product_type`, `communication_channel`, `conversation_style` | descriptive |
| `scenario` | JSON with the generation prompt, incl. `expected_outcome`, `expected_conversion_probability` |
| `probability_trajectory`, `customer_engagement`, `sales_effectiveness`, `conversation_flow`, `full_text`, `conversation_length` | generated annotations |

Mean 11.9 messages (8–30). The customer speaks first in 96%. About 21% of conversations have two
consecutive messages from the same speaker. `system` messages are markers such as
`--- a week later ---`.

**Outcome field.** `outcome`, balanced: 49.9% positive.

**Leakage risks.**
- `probability_trajectory` is a per-turn conversion probability, i.e. the answer to our task.
- `customer_engagement` and `sales_effectiveness` correlate 0.78 and 0.86 with the outcome.
- `scenario` contains `expected_outcome` and `expected_conversion_probability`.
- `conversation_flow` names the plot of the conversation (e.g. `initial_rejection`).
- The embeddings are computed from the full conversation.

None of these are used. Only `conversation` (text) and `outcome` (label) are read, plus company,
product, product type and channel as display metadata.

## Role and outcome mapping (implemented in Phase 1)

Two additions made while building the loaders: anything typed after an `accept` or `quit` in
Craigslist is dropped too, and conversations left with fewer than `data.min_turns` (2) turns are
removed (this affected one Craigslist dialogue).

| dataset | customer | agent | outcome = 1 | dropped before scoring | conversations kept |
|---|---|---|---|---|---|
| `craigslist` | buyer (`agent_turn == 0`) | seller | an `accept` act exists | all `offer` / `accept` / `reject` / `quit` turns; acts, prices, targets | 4,463 labelled (2,219 unlabelled excluded) |
| `p4g` | persuadee (`B4 == 1`) | persuader | persuadee `B6 > 0` (actual donation) | all of `full_info.csv` | 1,017 |
| `saas` | `customer` | `sales_rep` | `outcome == 1` | `system` messages; every annotation column | 20,000 sampled from 100,000 (seed 42, stratified by outcome) |

The Hugging Face train/validation/test split of Craigslist is ignored: its test split has no
labels. All three datasets get our own 70/15/15 split by conversation (seed 42, stratified by
outcome).
