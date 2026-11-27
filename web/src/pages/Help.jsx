import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/primitives';

const screens = [
  { name: 'Pipeline', path: '/', description: 'See summary metrics and leads grouped by status.' },
  { name: 'Leads', path: '/leads', description: 'Filter and rank conversations, then open one to explore its score curve, transcript and turning points.' },
  { name: 'Insights', path: '/insights', description: 'Explore how customer scores change after different sales moves.' },
  { name: 'Analyze', path: '/analyze', description: 'Score one or many conversations you paste or upload.' },
  { name: 'Live', path: '/live', description: 'Follow a conversation as it happens, or try the typed demo.' },
  { name: 'Annotate', path: '/annotate', description: 'Label customer turns and turning points for evaluation.' },
];

const statuses = [
  { term: 'Hot', definition: 'By default, the latest score is at least 70%. This takes priority over the trend, so a Hot lead can still be cooling.' },
  { term: 'Rising', definition: 'By default, the latest score is below 70%, and the recent trend is greater than +5 percentage points per message.' },
  { term: 'Cooling', definition: 'By default, the latest score is below 70%, and the recent trend is less than -5 percentage points per message.' },
  { term: 'Stalled', definition: 'By default, the latest score is below 70%, and the recent trend is between -5 and +5 percentage points per message, inclusive.' },
];

const glossary = [
  { term: 'Agent move', definition: 'An action by the sales side, such as offering a discount or addressing an objection.' },
  { term: 'Brier score', definition: 'The mean squared error of predicted probabilities; lower is better.' },
  { term: 'Calibration', definition: 'Adjusting scores so they better correspond to observed outcome rates.' },
  { term: 'Confidence interval', definition: 'A range expressing uncertainty around an estimated result.' },
  { term: 'Cue', definition: 'A labeled signal in a message, such as a price objection or a question about features.' },
  { term: 'Dataset', definition: 'A collection of conversations used to train or evaluate a model. SaaS conversations are synthetic; the other datasets are real conversations used as proxies for sales.' },
  { term: 'Faithfulness', definition: 'A check of how much the score changes when the message identified by an explanation is removed and the conversation is rescored.' },
  { term: 'Inclination score', definition: 'The model’s estimate of the likelihood of the dataset’s outcome, based on the conversation so far. It is not a guarantee of a purchase.' },
  { term: 'Leakage', definition: 'Information in the model input that reveals the outcome it is trying to predict.' },
  { term: 'Model', definition: 'A scoring method trained for a particular kind of conversation. Available models depend on the dataset and generated outputs.' },
  { term: 'Percentage points', definition: 'The difference between two percentage scores. A change from 40% to 45% is +5 points, not a 5% relative increase.' },
  { term: 'Prefix', definition: 'The conversation up to and including a particular turn.' },
  { term: 'ROC AUC', definition: 'A ranking measure: the chance that a randomly chosen positive outcome is scored above a randomly chosen negative outcome. 0.5 is chance.' },
  { term: 'Script insights', definition: 'The average score change on the customer’s next turn after a sales move. These are associations, not proof that the move caused the change.' },
  { term: 'Split', definition: 'A subset of the data used for training, validation or testing; conversations are kept together within one split.' },
  { term: 'Status', definition: 'A summary label, such as Hot, Rising, Cooling or Stalled, based on a lead’s score and recent changes.' },
  ...statuses,
  { term: 'Turn', definition: 'One message from one speaker.' },
  { term: 'Turning point', definition: 'A point where the score changes sharply or the curve shifts, depending on the selected method.' },
];

export default function Help() {
  const [query, setQuery] = useState('');
  const search = query.trim().toLowerCase();
  const results = glossary.filter(({ term, definition }) =>
    `${term} ${definition}`.toLowerCase().includes(search));

  return (
    <main className="mx-auto max-w-5xl px-5 py-8 md:px-8 md:py-12">
      <header className="mb-8">
        <h1 className="text-3xl font-extrabold tracking-tight md:text-4xl">Help</h1>
        <p className="mt-3 text-sm leading-6 text-muted-foreground">A quick guide to the workspace and its terminology.</p>
      </header>

      <section aria-labelledby="guide-title">
        <h2 id="guide-title" className="mb-4 text-2xl font-bold tracking-tight">Web app guide</h2>
        <Card>
          <CardHeader><CardTitle>Get started</CardTitle></CardHeader>
          <CardContent className="space-y-4 text-sm leading-6 text-muted-foreground">
            <p>Choose a dataset, model and split in the selector, then explore the Pipeline or Leads. Open a lead to inspect its transcript and score changes.</p>
            <p>To score your own chat, open <Link className="font-semibold text-primary underline-offset-4 hover:underline" to="/analyze">Analyze</Link>, paste text or upload a text, CSV or JSON file, check the speaker roles and conversation type, then select <strong>Score conversation</strong>. Try the sample if you do not have a chat ready. For multiple chats, select Many conversations; this requires a trained model.</p>
            <p>If the server or model is unavailable, Analyze labels its single-chat keyword result as a preview, not a trained-model score.</p>
          </CardContent>
        </Card>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          {screens.map(({ name, path, description }) => (
            <Card key={name}>
              <CardContent className="p-5">
                <h3 className="font-bold"><Link className="text-primary underline-offset-4 hover:underline" to={path}>{name}</Link></h3>
                <p className="mt-1 text-sm leading-6 text-muted-foreground">{description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
        <p className="mt-4 text-sm leading-6 text-muted-foreground">Use scores as a rough guide: models transfer poorly between conversation types, individual score changes can be unreliable, and insights are associations rather than proven causes.</p>
      </section>

      <section aria-labelledby="scores-title" className="mt-10">
        <h2 id="scores-title" className="mb-4 text-2xl font-bold tracking-tight">Understanding the numbers</h2>
        <Card>
          <CardContent className="space-y-4 p-5 text-sm leading-6 text-muted-foreground">
            <p><strong className="text-foreground">Score percentages:</strong> The percentage on a lead, message or score curve is the model’s estimated likelihood of the dataset’s outcome, based on the messages seen so far. The outcome may be a purchase, an accepted deal or a donation, depending on the dataset. The final score is the estimate after the last message; the Pipeline average is the average of those final scores. A score is not a guarantee, and preview scores are keyword-based rather than model predictions.</p>
            <p><strong className="text-foreground">+ and - points:</strong> Beside a message, these show the score change from the previous message, in percentage points. For example, 40% to 45% is <strong>+5 pts</strong>; 45% to 40% is <strong>-5 pts</strong>. Values are rounded for display. A turning-point marker uses the selected method: Score change compares consecutive messages, while Change point compares average scores across adjacent segments.</p>
            <p><strong className="text-foreground">Other percentages:</strong> In Insights, rise and drop percentages describe how often scores rose or fell after a move, not the chance of purchase. In evaluation mode, conversion and base rate show the share of conversations with a positive recorded outcome. A 95% confidence interval is a range showing uncertainty around an estimate, not a purchase probability.</p>
          </CardContent>
        </Card>
      </section>

      <section aria-labelledby="statuses-title" className="mt-10">
        <h2 id="statuses-title" className="mb-4 text-2xl font-bold tracking-tight">Lead statuses</h2>
        <Card>
          <CardContent className="space-y-4 p-5 text-sm leading-6 text-muted-foreground">
            <p>With the default settings, the app calculates the recent trend as the slope of a best-fit line through the last three scores (or fewer if the chat is shorter), in percentage points per message. If there is only one score, the trend is zero.</p>
            <dl className="space-y-3">
              {statuses.map(({ term, definition }) => (
                <div key={term}><dt className="font-bold text-foreground">{term}</dt><dd>{definition}</dd></div>
              ))}
            </dl>
            <p>These are automatic labels, not confirmed customer intentions. Thresholds can be configured, and previously exported leads keep the settings used when they were generated. Rounded scores can appear to be on a boundary even when the underlying score is not.</p>
          </CardContent>
        </Card>
      </section>

      <section aria-labelledby="glossary-title" className="mt-10">
        <h2 id="glossary-title" className="text-2xl font-bold tracking-tight">Project glossary</h2>
        <label htmlFor="glossary-search" className="mt-4 block text-sm font-semibold">Search terms and definitions</label>
        <input id="glossary-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)}
          placeholder="Search the glossary" className="mt-2 h-10 w-full max-w-md rounded-xl border border-input bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring" />
        <div aria-live="polite" className="mt-4">
          {results.length ? (
            <dl className="divide-y divide-border rounded-2xl border border-border bg-card">
              {results.map(({ term, definition }) => (
                <div key={term} className="px-5 py-4">
                  <dt className="font-bold">{term}</dt>
                  <dd className="mt-1 text-sm leading-6 text-muted-foreground">{definition}</dd>
                </div>
              ))}
            </dl>
          ) : <p className="text-sm text-muted-foreground">No matching glossary terms.</p>}
        </div>
      </section>
    </main>
  );
}
