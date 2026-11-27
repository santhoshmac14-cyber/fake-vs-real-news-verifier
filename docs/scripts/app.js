const input = document.querySelector('#news-text');
const button = document.querySelector('#analyze-button');
const loading = document.querySelector('#loading');
const errorBox = document.querySelector('#error');
const results = document.querySelector('#results');
const count = document.querySelector('#char-count');
let modelData;

const topicWords = {
  Politics: ['president','congress','election','senate','government','politician','parliament','campaign','vote','white house'],
  'World News': ['war','international','border','foreign','israel','russia','ukraine','china','united nations','overseas','diplomat'],
  Business: ['market','stock','company','business','economy','trade','profit','bank','financial','jobs','price','oil'],
  Health: ['health','hospital','disease','vaccine','medical','doctor','patient','virus','medicine','clinical','wellness'],
  Technology: ['technology','tech','internet','software','google','apple','artificial intelligence','computer','cyber','smartphone','app'],
  Science: ['research','scientist','science','study','laboratory','space','astronomy','discovery','experiment','nasa'],
  Sports: ['sports','football','basketball','game','team','player','coach','league','match','tournament','championship'],
  Entertainment: ['film','movie','actor','actress','music','celebrity','concert','television','singer','album','hollywood'],
  Environment: ['climate','environment','carbon','wildlife','conservation','pollution','renewable','forest','drought','flood']
};

fetch('model.json')
  .then((response) => {
    if (!response.ok) throw new Error('Could not load the model file.');
    return response.json();
  })
  .then((data) => {
    modelData = data;
    modelData.vocabulary = new Map(data.features.map(([term, idf, weight]) => [term, { idf, weight }]));
    modelData.stopWords = new Set(data.stop_words);
    loading.textContent = 'Classifier ready. Paste text to analyze.';
  })
  .catch(() => {
    loading.textContent = 'The model file did not load. Run this page from a local web server or GitHub Pages.';
    button.disabled = true;
  });

input.addEventListener('input', () => {
  count.textContent = `${input.value.length.toLocaleString()} / 30,000 characters`;
});
button.addEventListener('click', analyzeText);

function tokenize(text) {
  const tokens = text.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || [];
  return tokens.filter((token) => !modelData.stopWords.has(token));
}

function classify(text) {
  const tokens = tokenize(text);
  const frequencies = new Map();
  for (const token of tokens) frequencies.set(token, (frequencies.get(token) || 0) + 1);
  for (let index = 0; index < tokens.length - 1; index += 1) {
    const phrase = `${tokens[index]} ${tokens[index + 1]}`;
    frequencies.set(phrase, (frequencies.get(phrase) || 0) + 1);
  }

  let lengthSquared = 0;
  const present = [];
  for (const [term, frequency] of frequencies) {
    const feature = modelData.vocabulary.get(term);
    if (!feature) continue;
    const value = frequency * feature.idf;
    present.push({ term, value, feature });
    lengthSquared += value * value;
  }
  const norm = Math.sqrt(lengthSquared) || 1;
  let decision = modelData.intercept;
  for (const item of present) decision += (item.value / norm) * item.feature.weight;
  const realProbability = 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, decision))));
  const prediction = realProbability >= 0.5 ? 'Real' : 'Fake';
  const direction = prediction === modelData.classes[1] ? 1 : -1;
  const keyTerms = present
    .map((item) => ({ term: item.term, impact: (item.value / norm) * item.feature.weight * direction }))
    .filter((item) => item.impact > 0)
    .sort((a, b) => b.impact - a.impact)
    .slice(0, 6);
  return { prediction, realProbability, keyTerms };
}

function suggestTopic(text) {
  const normalized = text.toLowerCase();
  let selected = 'General / Unclear';
  let highest = 0;
  for (const [topic, words] of Object.entries(topicWords)) {
    let score = 0;
    for (const word of words) {
      if (word.includes(' ')) score += normalized.split(word).length - 1;
      else score += (normalized.match(new RegExp(`\\b${word}\\b`, 'g')) || []).length;
    }
    if (score > highest) { highest = score; selected = topic; }
  }
  return selected;
}

function analyzeText() {
  errorBox.hidden = true;
  const text = input.value.trim();
  if (!text) {
    errorBox.textContent = 'Paste a headline or article before analyzing.';
    errorBox.hidden = false;
    return;
  }
  if (!modelData) return;
  const result = classify(text);
  const fakeScore = 1 - result.realProbability;
  document.querySelector('#prediction').textContent = result.prediction;
  document.querySelector('#topic').textContent = suggestTopic(text);
  document.querySelector('#fake-label').textContent = `${(fakeScore * 100).toFixed(1)}%`;
  document.querySelector('#real-label').textContent = `${(result.realProbability * 100).toFixed(1)}%`;
  document.querySelector('#fake-bar').style.width = `${fakeScore * 100}%`;
  document.querySelector('#real-bar').style.width = `${result.realProbability * 100}%`;
  const terms = document.querySelector('#terms');
  terms.replaceChildren();
  if (result.keyTerms.length) {
    for (const item of result.keyTerms) {
      const chip = document.createElement('span');
      chip.className = 'term-chip';
      chip.textContent = item.term;
      terms.append(chip);
    }
  } else {
    const note = document.createElement('span');
    note.className = 'term-empty';
    note.textContent = 'No learned words or phrases from the model vocabulary were found in this text.';
    terms.append(note);
  }
  results.hidden = false;
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
