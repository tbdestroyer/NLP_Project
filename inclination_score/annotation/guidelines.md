# Annotation guidelines

These labels are the human reference for the Inclination Score. We use them to check whether the
model's score moves in the same direction a person would say, and whether its turning points
match the ones a person sees.

## Setup

1. Run `make all` once, then `make app` (`.\make.ps1 app` on Windows).
2. Open **Annotate** in the left menu.
3. Type your annotator name in the sidebar. Use the same name every time; your labels are saved
   to `annotation/labels/<name>.jsonl`.
4. There are 60 conversations: 20 from each dataset, all from the test split. The list is fixed
   in `annotation/sample.json`, so every annotator labels the same ones.

Work alone. Do not look at the Conversation View for a conversation before you have labelled it,
and do not discuss individual conversations with other annotators until everyone is done.

## What "inclined" means

How likely the customer seems, at this moment, to end up saying yes:

| dataset | customer | "yes" means |
|---|---|---|
| CraigslistBargains | the buyer | agreeing a deal with the seller |
| Persuasion for Good | the person being asked | actually donating to the charity |
| SaaS (synthetic) | the prospect | buying or signing up for the product |

## The two labels

### 1. Direction, on every customer turn

Compared with just before this message, is the customer now:

- **More inclined:** the message moves them toward yes. Interest, a question about how to buy,
  agreement, a commitment, accepting a price.
- **Same:** nothing changed. Small talk, a neutral question, an acknowledgement.
- **Less inclined:** the message moves them away from yes. An objection, doubt, a refusal,
  stalling, mentioning another option, a much shorter and cooler reply.

For the customer's first turn, compare with a neutral starting point: someone who has just
joined and has not shown any leaning.

Agent turns do not get a direction. The effect of what the agent says shows up in your label for
the customer's next turn.

### 2. Turning point, on any turn

Tick it on a turn where the conversation clearly changes course: you would point to this message
if asked "where was this customer won or lost?". It can be a customer turn or an agent turn.
Most conversations have none, one or two. If you are ticking it on a third of the turns, you are
using it too freely.

## Rules of thumb

- **Judge the change, not the level.** A keen customer who says "ok" is *same*, not *more*.
- **Use only what has been said so far.** Read top to bottom and decide each turn before reading
  the next one. Do not go back and change earlier labels because of how it ended.
- **Read the meaning, not the keywords.** "That's a lot" about a price is an objection. "That's a
  lot of features" is not.
- **Politeness is not interest.** "Thanks, I'll think about it" is usually *less*, not *more*.
- **A question about price is not an objection.** "How much is it?" is *same* or mildly *more*.
  "Can you do better than that?" is an objection, but in a negotiation it also shows the buyer is
  still engaged: label it *same* unless the tone is clearly cooling.
- **When you cannot decide, choose _same_.** Do not force a direction.
- **Stated intent counts as stated.** If the customer says "I'll donate $1", label it *more* even
  if you suspect they will not follow through.

## Worked examples

These are illustrative conversations written for these guidelines, not taken from the datasets.

### Example 1: negotiation

| turn | speaker | message | direction | turning point |
|---|---|---|---|---|
| 0 | agent | Hi, the bike is still available. Asking $200. | | |
| 1 | customer | Looks nice. How old is it? | more | |
| 2 | agent | Two years, barely ridden. New tyres last month. | | |
| 3 | customer | Hmm, $200 is more than I wanted to spend. | less | |
| 4 | agent | I could do $170 if you pick it up today. | | ✔ |
| 5 | customer | $170 works. I can come at 5. | more | |

Turn 1 is *more*: a question about the item shows real interest. Turn 3 is *less*: a price
objection. Turn 4 is the turning point, because the discount is what rescues the deal. Turn 5 is
*more*, but it is the result of turn 4, so it is not marked as a second turning point.

### Example 2: donation

| turn | speaker | message | direction | turning point |
|---|---|---|---|---|
| 0 | agent | Have you heard of Save the Children? | | |
| 1 | customer | Yes, I think so. | same | |
| 2 | agent | They help children in war zones. Would you donate some of your task payment? | | |
| 3 | customer | How do I know the money actually gets there? | less | ✔ |
| 4 | agent | Over 85% of donations go directly to programmes. | | |
| 5 | customer | Ok. | same | |
| 6 | agent | Even $0.50 would help. | | |
| 7 | customer | Maybe another time. | less | |

Turn 1 is *same*: an acknowledgement, no leaning. Turn 3 is *less* and a turning point: distrust
appears and the conversation never recovers. Turn 5 is *same*: "Ok" accepts the fact but shows no
movement toward donating. Turn 7 is *less*: a polite refusal.

### Example 3: sales

| turn | speaker | message | direction | turning point |
|---|---|---|---|---|
| 0 | customer | We're drowning in manual reports. Does your tool automate them? | more | |
| 1 | agent | Yes, reports build themselves from your data every morning. | | |
| 2 | customer | Nice. Does it connect to Salesforce? | more | |
| 3 | agent | It does. Pricing is $900 per month. | | |
| 4 | customer | Oh. We're already paying for a BI tool, I'd have to justify that. | less | ✔ |
| 5 | agent | Most teams drop the BI tool within a quarter. I can share a case study. | | |
| 6 | customer | Send it over and I'll take it to my manager. | same | |

Turn 0 is *more* compared with a neutral start: the customer opens with a need and a product
question. Turn 4 is *less* and a turning point: a price objection plus an existing alternative.
Turn 6 is *same*: the customer stays engaged but has postponed the decision, which neither helps
nor hurts compared with turn 4.

## Saving

Press **Save and go to the next unlabelled**. Every customer turn needs a direction before the
form will save. You can go back and re-save a conversation; the latest save is the one used.
