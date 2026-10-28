// Reads uploaded or pasted transcripts into conversations: [{ title, turns: [{ speaker, role?, text }] }].
// A turn has `role` only when the file says so ('agent' or 'customer'); otherwise roles are guessed
// from speaker names (see roleGuesser) and the user can correct them.

export function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"' && quoted && text[i + 1] === '"') {
      field += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === ',' && !quoted) {
      row.push(field);
      field = '';
    } else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && text[i + 1] === '\n') i += 1;
      row.push(field);
      if (row.some((value) => value.trim())) rows.push(row);
      row = [];
      field = '';
    } else {
      field += char;
    }
  }
  row.push(field);
  if (row.some((value) => value.trim())) rows.push(row);
  return rows;
}

const ROLES = ['agent', 'customer'];
const ID_COLUMNS = ['conversation_id', 'conversation', 'chat_id', 'call_id', 'thread_id', 'transcript_id', 'id'];
const TITLE_COLUMNS = ['title', 'name', 'account', 'company', 'deal'];
const SEPARATOR = /^\s*(-{3,}|={3,})\s*$/;
const TITLE_LINE = /^\s*#{1,6}\s*(.+?)\s*$/;

const declaredRole = (value) => (ROLES.includes(String(value ?? '').trim().toLowerCase()) ? String(value).trim().toLowerCase() : undefined);

function toTurn(record) {
  const speaker = String(record.speaker ?? record.role ?? record.author ?? record.from ?? 'Unknown').trim() || 'Unknown';
  return { speaker, role: declaredRole(record.role) ?? declaredRole(record.speaker), text: String(record.text ?? record.message ?? record.content ?? '').trim() };
}

function isCsvHeader(line) {
  const columns = line.split(',').map((column) => column.trim().replace(/^"|"$/g, '').toLowerCase());
  return columns.some((column) => ['speaker', 'role'].includes(column))
    && columns.some((column) => ['text', 'message', 'content'].includes(column));
}

function parseLines(text) {
  const turns = [];
  for (const line of text.split(/\r?\n/).filter((l) => l.trim())) {
    const m = line.match(/^\s*([^:]{1,30}):\s*(.+)$/);
    if (m) turns.push({ speaker: m[1].trim(), text: m[2].trim() });
    else if (turns.length) turns[turns.length - 1].text += ` ${line.trim()}`;
  }
  return turns;
}

function fromCsv(text) {
  const rows = parseCsv(text);
  const headers = rows.shift().map((header) => header.trim().toLowerCase());
  const column = (names) => headers.findIndex((header) => names.includes(header));
  const speakerColumn = headers.indexOf('speaker');
  const roleColumn = headers.indexOf('role');
  const textColumn = column(['text', 'message', 'content']);
  const idColumn = column(ID_COLUMNS);
  const titleColumn = column(TITLE_COLUMNS);
  const groups = new Map();
  for (const row of rows) {
    const id = idColumn >= 0 ? row[idColumn]?.trim() || '' : '';
    if (!groups.has(id)) groups.set(id, { title: titleColumn >= 0 ? row[titleColumn]?.trim() || id : id, turns: [] });
    const role = roleColumn >= 0 ? row[roleColumn] : undefined;
    groups.get(id).turns.push(toTurn({ speaker: (speakerColumn >= 0 && row[speakerColumn]?.trim()) || role, role, text: row[textColumn] }));
  }
  return [...groups.values()];
}

function fromJson(decoded) {
  const list = Array.isArray(decoded) ? decoded : decoded.conversations ?? (decoded.turns || decoded.messages ? [decoded] : null);
  if (!Array.isArray(list)) return [];
  // A list of turns is one conversation; a list of objects with turns/messages is many.
  const many = list.some((item) => Array.isArray(item?.turns) || Array.isArray(item?.messages));
  const conversations = many ? list : [{ turns: list }];
  return conversations.map((item) => ({
    title: String(item.title ?? item.name ?? item.conversation_id ?? item.id ?? '').trim(),
    turns: (item.turns ?? item.messages ?? []).map(toTurn),
  }));
}

/**
 * Every conversation in a file or pasted text. Accepts:
 * - "Speaker: message" lines; several conversations are separated by a line of --- (or ===), and
 *   an optional "# Title" line starts a conversation;
 * - CSV with speaker/role and text/message columns, plus an optional conversation_id column
 *   (one conversation per id) and title column;
 * - JSON: a list of turns, or a list of { title, turns } (also { conversations: [...] }).
 */
export function parseTranscripts(raw, fallbackTitle = 'Conversation') {
  const text = raw.trim();
  if (!text) return [];
  let conversations;
  if (text.startsWith('[') || text.startsWith('{')) {
    try {
      conversations = fromJson(JSON.parse(text));
    } catch {
      return [];
    }
  } else if (isCsvHeader(text.split(/\r?\n/, 1)[0])) {
    conversations = fromCsv(text);
  } else {
    conversations = [];
    let chunk = [];
    const flush = () => {
      const lines = chunk.filter((line) => line.trim());
      chunk = [];
      if (!lines.length) return;
      const heading = lines[0].match(TITLE_LINE);
      conversations.push({ title: heading ? heading[1] : '', turns: parseLines((heading ? lines.slice(1) : lines).join('\n')) });
    };
    for (const line of text.split(/\r?\n/)) {
      if (SEPARATOR.test(line)) flush();
      else chunk.push(line);
    }
    flush();
  }
  const kept = conversations
    .map((conversation) => ({ ...conversation, turns: conversation.turns.filter((turn) => turn.text) }))
    .filter((conversation) => conversation.turns.length);
  return kept.map((conversation, index) => ({
    ...conversation,
    title: conversation.title || (kept.length === 1 ? fallbackTitle : `${fallbackTitle} ${index + 1}`),
  }));
}

/** The turns of one conversation (the first one, if the text holds several). */
export function parseConversation(raw) {
  return parseTranscripts(raw)[0]?.turns ?? [];
}

export const AGENT_HINT = /^(rep|agent|seller|sales|persuader|bot|assistant|support|account exec|ae\b|sdr|bdr)/i;
const CUSTOMER_HINT = /^(customer|client|buyer|prospect|lead|user|persuadee|caller|visitor)/i;

/**
 * Returns role(speaker) for one conversation: the user's choice, else the role the file declared,
 * else a guess from the name; with no hint at all, whoever speaks first is the sales side.
 */
export function roleGuesser(turns, overrides = {}) {
  const speakers = [...new Set(turns.map((turn) => turn.speaker))];
  const declared = Object.fromEntries(turns.filter((turn) => turn.role).map((turn) => [turn.speaker, turn.role]));
  const hasAgentHint = speakers.some((speaker) => AGENT_HINT.test(speaker) || declared[speaker] === 'agent');
  return (speaker) => overrides[speaker] ?? declared[speaker]
    ?? (AGENT_HINT.test(speaker) ? 'agent'
      : CUSTOMER_HINT.test(speaker) || hasAgentHint ? 'customer'
        : speaker === speakers[0] ? 'agent' : 'customer');
}

/** Turns with every role filled in, ready for POST /api/analyze. */
export function withRoles(turns, overrides) {
  const roleOf = roleGuesser(turns, overrides);
  return turns.map((turn) => ({ role: roleOf(turn.speaker), text: turn.text, speaker: turn.speaker }));
}

export const hasBothSides = (turns) => {
  const roles = new Set(turns.map((turn) => turn.role));
  return roles.has('agent') && roles.has('customer');
};
