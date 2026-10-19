const seedEl = document.getElementById('seed');
const tempEl = document.getElementById('temp');
const tempVal = document.getElementById('temp-val');
const goBtn = document.getElementById('go');
const result = document.getElementById('result');
const seqLabel = document.getElementById('seq-label');

tempEl.oninput = () => tempVal.textContent = tempEl.value;

function paintMap(canvas, matrix, chars) {
  // Heatmap: cell (row, col) = how much position `row` attends to `col`.
  const ctx = canvas.getContext('2d');
  const T = matrix.length;
  const cell = canvas.width / T;
  let max = 0.0001;
  for (const row of matrix) for (const v of row) max = Math.max(max, v);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < T; i++)
    for (let j = 0; j < T; j++) {
      const a = matrix[i][j] / max;
      ctx.fillStyle = `rgba(122,162,255,${a.toFixed(3)})`;
      ctx.fillRect(j * cell, i * cell, cell, cell);
    }
  // Axis labels: the characters themselves (space shown as middot).
  ctx.fillStyle = '#9aa3b2';
  ctx.font = '9px monospace';
  for (let i = 0; i < T; i++) {
    const ch = chars[i] === ' ' ? '·' : (chars[i] === '\n' ? '⏎' : chars[i]);
    ctx.fillText(ch, 1, i * cell + cell * 0.75);
    ctx.fillText(ch, i * cell + 2, 9);
  }
}

goBtn.onclick = async () => {
  goBtn.disabled = true;
  result.textContent = 'thinking…'; result.className = 'result quiet';
  try {
    const r = await fetch('/api/generate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({seed: seedEl.value, temperature: parseFloat(tempEl.value)})
    });
    const data = await r.json();
    result.textContent = data.text || '(empty - try again)';
    result.className = 'result';
    for (let i = 0; i < 2; i++)
      paintMap(document.getElementById('map' + i), data.attention[i], data.sequence);
    seqLabel.textContent = 'sequence: ' + JSON.stringify(data.sequence);
  } finally {
    goBtn.disabled = false;
  }
};
