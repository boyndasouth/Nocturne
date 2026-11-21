// Facts are stored in the third person ("The user lives in Dallas."), which is what the
// models read. People see them addressed to themselves ("You live in Dallas."), and what
// they type when correcting one is turned back into the stored form before saving.

const SUBJECT_BEFORE = /(^|[.:;!?]\s+|,\s+|\b(?:and|but|so|because|when|while|that|since|after|before|until|if|or|now)\s+)$/i;
const ADVERBS = new Set([
  "also", "usually", "often", "never", "always", "just", "still", "really", "now", "already",
  "currently", "sometimes", "rarely", "only", "recently", "no", "not", "even", "mostly",
]);
const TO_YOU: Record<string, string> = {
  is: "are", was: "were", has: "have", does: "do", "doesn't": "don't", "isn't": "aren't",
  "wasn't": "weren't", "hasn't": "haven't", "is not": "are not",
};
const TO_USER: Record<string, string> = {
  are: "is", were: "was", have: "has", do: "does", "don't": "doesn't", "aren't": "isn't",
  "weren't": "wasn't", "haven't": "hasn't", am: "is", "'m": "is", "'ve": "has", "'re": "is",
};
const PAST_OR_MODAL = new Set([
  "can", "could", "will", "would", "should", "might", "may", "must", "shall", "did", "didn't",
  "can't", "won't", "wouldn't", "couldn't", "shouldn't", "had", "ran", "went", "got", "made",
  "took", "began", "came", "saw", "knew", "thought", "felt", "left", "lost", "won", "bought",
  "sold", "met", "paid", "said", "told", "found", "gave", "kept", "spent", "wrote", "drove",
  "flew", "grew", "became", "broke", "chose", "ate", "drank", "swam", "sang", "taught", "built",
  "used", "needs", "loves",
]);

/** "lives" → "live", "studies" → "study", "watches" → "watch". */
function unconjugate(verb: string): string {
  const lower = verb.toLowerCase();
  if (TO_YOU[lower]) return TO_YOU[lower];
  if (/ed$/.test(lower) || PAST_OR_MODAL.has(lower)) return verb;
  if (/[^aeiou]ies$/.test(lower)) return verb.slice(0, -3) + "y";
  if (/(ch|sh|ss|x|z|o)es$/.test(lower)) return verb.slice(0, -2);
  if (/[^su]s$/.test(lower)) return verb.slice(0, -1);
  return verb;
}

/** "live" → "lives", "study" → "studies". */
function conjugate(verb: string): string {
  const lower = verb.toLowerCase();
  if (TO_USER[lower]) return TO_USER[lower];
  if (/ed$/.test(lower) || PAST_OR_MODAL.has(lower) || /s$/.test(lower)) return verb;
  if (/[^aeiou]y$/.test(lower)) return verb.slice(0, -1) + "ies";
  if (/(ch|sh|x|z|o)$/.test(lower)) return verb + "es";
  return verb + "s";
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** Stored third person → addressed to the reader. */
export function toYou(text: string): string {
  let out = "";
  let rest = text;
  const re = /\b(the user)('s)?(\s+)([\w']+)?(\s+[\w']+)?/i;
  for (let m = re.exec(rest); m; m = re.exec(rest)) {
    const before = rest.slice(0, m.index);
    out += before;
    const atStart = (out.trim() === "" || /[.:;!?]\s*$/.test(out));
    const subject = SUBJECT_BEFORE.test(out);
    const [whole, , possessive, space, w1 = "", w2 = ""] = m;
    let rep: string;
    if (possessive) {
      rep = `your${space}${w1}${w2}`;
    } else if (!subject) {
      rep = `you${space}${w1}${w2}`;
    } else if (w1 && ADVERBS.has(w1.toLowerCase()) && w2) {
      rep = `you${space}${w1} ${unconjugate(w2.trim())}`;
    } else {
      rep = `you${space}${w1 ? unconjugate(w1) : ""}${w2}`;
    }
    out += atStart ? cap(rep) : rep;
    rest = rest.slice(m.index + whole.length);
  }
  return out + rest;
}

/** What a person types ("I live in Plano", "You live in Plano") → the stored third person. */
export function toStored(text: string): string {
  let t = " " + text.trim() + " ";
  t = t.replace(/\b(I|you)'(m|re|ve)\b/gi, (_, __, c: string) => `the user ${TO_USER["'" + c.toLowerCase()]}`);
  t = t.replace(/\b(my|your|mine|yours)\b/gi, "the user's");
  t = t.replace(/\b(me|myself|yourself)\b/gi, "the user");
  t = t.replace(/\b(I|you)(\s+)([\w']+)(\s+[\w']+)?/gi, (whole, who: string, sp: string, w1: string, w2 = "") => {
    if (/^you$/i.test(who) && !SUBJECT_BEFORE.test(t.slice(0, t.indexOf(whole)).trimStart())) {
      return `the user${sp}${w1}${w2}`;
    }
    if (ADVERBS.has(w1.toLowerCase()) && w2) return `the user${sp}${w1} ${conjugate(w2.trim())}`;
    return `the user${sp}${conjugate(w1)}${w2}`;
  });
  t = t.trim();
  t = t.replace(/(^|[.!?]\s+)the user/g, (_, p: string) => `${p}The user`);
  if (!/[.!?]$/.test(t)) t += ".";
  return cap(t);
}
