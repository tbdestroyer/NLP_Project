# Data notes (Phase 0 inspection)

Produced by `make download` and `make inspect` on 2026-10-07. The full printed schemas and example
records are in `outputs/data_inspection.txt`. Everything below was read from the real files, not
from the spec.

## 1. CraigslistBargains (`craigslist`) — real humans

Updated on 2026-10-19 when the source was switched to the original release.

**Source used.** The original CoCoA release on the
[CodaLab worksheet](https://worksheets.codalab.org/worksheets/0x453913e76b65495d8b9730d41c7e0a0c/):
`train.json`, `dev.json` and `test.json`. CodaLab's TLS certificate has expired, so `make download`
cannot fetch them. Open the worksheet in a browser, open each of the three files, save it under
that name in `data/raw/craigslist/`, then run `make data`; `make download` stops with a message
if a file is missing.

The first version used Hugging Face's parquet conversion (`stanfordnlp/craigslist_bargains`,
branch `refs/convert/parquet`). It is the same 6,682 dialogues in the same order (HF train /
validation / test = `train.json` / `dev.json` / `test.json`, checked row by row), but a third of
them lost their dialogue acts and with them their outcome (see below).

**Structure.** 6,682 dialogues (train 5,247 / dev 597 / test 838), one JSON object each:

| field | content |
|---|---|
| `scenario.kbs` | one entry per agent: `personal` (`Role`, private `Target` price, `Bottomline`) and `item` (`Title`, `Category`, `Price` = listing price, `Description`, `Images`). Index 0 is always the buyer |
| `events` | one per turn: `agent` (0 or 1, an index into `kbs`), `action` (`message`, `offer`, `accept`, `reject`, `quit`), `data` (the text of a message, the price of an offer, else empty) |
| `outcome` | `reward` (1 = deal, 0 = no deal) and the final `offer` |

Mean 9.2 events per dialogue (min 2, max 46). Buyer speaks first in 57%. Six categories:
furniture, housing, bike, car, electronics, phone.

**Outcome field.** `outcome.reward`, present for every dialogue: 74.9% are deals.

**Recovered dialogues.** 2,219 dialogues have no dialogue acts in the HF copy (**all 838 of its
test split**, 1,237 of train and 144 of validation), so their outcome was unknown there and they
were excluded. Here they are labelled. Their deal rate is 52.9%, against 85.8% for the 4,463
dialogues that had acts, so the excluded third was mostly where the negatives were. The ids of
the 2,219 are listed in `data/craigslist_recovered.json`.

For the 4,463 dialogues that had acts, this loader gives exactly the same records as the HF one
(turns, outcome, title and metadata; checked one by one), so their ids and split do not change.

All 6,682 dialogues by their strongest action event:

| has | count | outcome (`reward`) |
|---|---|---|
| `accept` | 5,006 (74.9%) | 1, except one: an `accept` clicked with no offer on the table, scored 0 |
| `quit` | 579 | 0 |
| `reject` | 463 | 0 |
| an `offer` nobody answered | 38 | 0 |
| no action event at all (abandoned) | 596 | 0 |

`reward` is used as the outcome. On the 4,463 dialogues of the HF copy it always agrees with the
earlier rule ("an `accept` act exists").

**Leakage risks.**
- Action events (`offer`, `accept`, `reject`, `quit`): never text, and the `offer` carries the
  final deal price. All are dropped from the turn table.
- Action events are **not always at the end**: in 1,491 dialogues people keep typing after an
  action (for example, an offer is rejected and bargaining continues). Those messages are kept.
- `outcome` (reward and final offer): label only.
- `kbs[].personal.Target` (private target prices): never read.

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
removed (this affected one Craigslist dialogue of the HF copy, and 351 of the full release: 133
with no message at all and 218 with one).

| dataset | customer | agent | outcome = 1 | dropped before scoring | conversations kept |
|---|---|---|---|---|---|
| `craigslist` | buyer (`kbs[0]`) | seller | `outcome.reward == 1` | all `offer` / `accept` / `reject` / `quit` events; outcome, prices, targets | 6,331 of 6,682 |
| `p4g` | persuadee (`B4 == 1`) | persuader | persuadee `B6 > 0` (actual donation) | all of `full_info.csv` | 1,017 |
| `saas` | `customer` | `sales_rep` | `outcome == 1` | `system` messages; every annotation column | 20,000 sampled from 100,000 (seed 42, stratified by outcome) |

The release's own train/dev/test split of Craigslist is ignored (the HF copy had no labels for
its test split). All three datasets get our own 70/15/15 split by conversation (seed 42,
stratified by outcome). The recovered Craigslist dialogues are split on their own, the same way,
so the 4,462 used before keep their assignment and the annotation sample stays in test.
