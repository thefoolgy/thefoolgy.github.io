(async () => {
  const response = await fetch('./cases.json?v=20260802-1');
  if (!response.ok) throw new Error(`Failure dataset request returned ${response.status}`);
  const data = await response.json();
  if (!Array.isArray(data.cases)) throw new Error('Failure dataset is invalid');

  const list = document.getElementById('case-list');
  const count = document.getElementById('result-count');
  const empty = document.getElementById('empty-state');
  const search = document.getElementById('search-input');
  const englishToggle = document.getElementById('english-toggle');
  const reset = document.getElementById('reset-button');
  const filterButtons = [...document.querySelectorAll('[data-filter]')];
  let activeFilter = 'all';

  const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  })[char]);

  const f1 = (value) => Number(value).toFixed(3);

  const originalDetails = (label, paragraphs) => `
    <details class="original">
      <summary>${escapeHtml(label)}</summary>
      ${paragraphs.map((text) => `<p>${escapeHtml(text)}</p>`).join('')}
    </details>`;

  const modelCard = (name, model) => {
    const queries = model.generated_queries.length
      ? `<ul class="query-list">${model.generated_queries.map((query) => `<li>${escapeHtml(query)}</li>`).join('')}</ul>`
      : '';
    return `
      <article class="model-card ${escapeHtml(name)}">
        <header><b>${escapeHtml(model.label)}</b><span>F1 ${f1(model.token_f1)} · ${model.num_steps} step</span></header>
        <p class="prediction-zh">${escapeHtml(model.prediction_zh)}</p>
        <p class="prediction-en">${escapeHtml(model.prediction)}</p>
        ${queries}
      </article>`;
  };

  const evidenceCard = (evidence) => `
    <article class="evidence-card">
      <h4>${escapeHtml(evidence.title_zh)} <small>${escapeHtml(evidence.title)}</small></h4>
      <ul>${evidence.facts_zh.map((fact) => `<li>${escapeHtml(fact)}</li>`).join('')}</ul>
      ${originalDetails('查看英文金标句', evidence.facts)}
    </article>`;

  const caseCard = (item) => `
    <article class="case-card" data-class="${escapeHtml(item.review_class)}" id="${escapeHtml(item.question_id)}">
      <header class="case-head">
        <span class="case-index">#${String(item.index).padStart(2, '0')}</span>
        <div class="case-title">
          <h3>${escapeHtml(item.question_zh)}</h3>
          ${originalDetails('查看英文问题', [item.question])}
        </div>
        <div class="case-meta">
          <span class="chip ${escapeHtml(item.review_class)}">${escapeHtml(item.review_class_zh)}</span>
          <span class="chip id">${escapeHtml(item.question_id)}</span>
        </div>
      </header>
      <div class="case-body">
        <div class="answer-row">
          <span>Gold answer</span>
          <strong>${escapeHtml(item.ground_truth_zh)}<small>${escapeHtml(item.ground_truth)}</small></strong>
        </div>
        <div class="diagnosis"><b>人工复核</b><p>${escapeHtml(item.diagnosis_zh)}</p></div>
        <div class="subhead">三种条件下的答案</div>
        <div class="model-grid">
          ${Object.entries(item.models).map(([name, model]) => modelCard(name, model)).join('')}
        </div>
        <div class="subhead">Gold supporting evidence</div>
        <div class="evidence-grid">${item.evidence.map(evidenceCard).join('')}</div>
      </div>
    </article>`;

  const searchText = (item) => JSON.stringify(item).toLowerCase();
  const indexed = data.cases.map((item) => ({ item, text: searchText(item) }));

  const render = () => {
    const query = search.value.trim().toLowerCase();
    const visible = indexed.filter(({ item, text }) => {
      const classMatch = activeFilter === 'all' || item.review_class === activeFilter;
      return classMatch && (!query || text.includes(query));
    }).map(({ item }) => item);
    list.innerHTML = visible.map(caseCard).join('');
    count.textContent = String(visible.length);
    empty.hidden = visible.length !== 0;
    document.querySelectorAll('details.original').forEach((details) => {
      details.open = englishToggle.checked;
    });
  };

  const setFilter = (filter) => {
    activeFilter = filter;
    filterButtons.forEach((button) => button.classList.toggle('active', button.dataset.filter === filter));
    render();
  };

  filterButtons.forEach((button) => button.addEventListener('click', () => setFilter(button.dataset.filter)));
  document.querySelectorAll('[data-filter-shortcut]').forEach((button) => button.addEventListener('click', () => {
    setFilter(button.dataset.filterShortcut);
    document.getElementById('catalog-title').scrollIntoView({ behavior: 'smooth' });
  }));
  search.addEventListener('input', render);
  englishToggle.addEventListener('change', () => {
    document.querySelectorAll('details.original').forEach((details) => {
      details.open = englishToggle.checked;
    });
  });
  reset.addEventListener('click', () => {
    search.value = '';
    englishToggle.checked = false;
    setFilter('all');
  });

  render();
})().catch((error) => {
  console.error(error);
  const list = document.getElementById('case-list');
  if (list) list.innerHTML = '<div class="empty-state"><strong>数据加载失败</strong><p>请刷新页面后重试。</p></div>';
});
