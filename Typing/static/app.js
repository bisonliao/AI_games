const app = document.querySelector("#app");
const capture = document.querySelector("#typing-capture");

const state = {
  view: "home",
  lessons: [],
  exercise: null,
  exerciseParams: null,
  typed: [],
  startedAt: null,
  finishedAt: null,
  composing: false,
};

const categoryCards = [
  {
    id: "beginner",
    title: "初阶",
    meta: "9 个课时",
    body: "练习基本指法：左右手手指和大拇指。",
  },
  {
    id: "intermediate",
    title: "中阶",
    meta: "英文文章",
    body: "约 100 个单词，练习英文句子和标点。",
  },
  {
    id: "advanced",
    title: "高阶",
    meta: "中文文章",
    body: "约 300 个汉字，练习中文输入和中文标点。",
  },
];

function normalizeHash() {
  return window.location.hash.replace(/^#\/?/, "");
}

async function loadLessons() {
  const response = await fetch("/api/lessons");
  const data = await response.json();
  state.lessons = data.lessons;
}

function setRoute(route) {
  window.location.hash = route;
}

function render() {
  const route = normalizeHash();
  if (!route || route === "home") {
    state.view = "home";
    renderHome();
    return;
  }
  if (route === "beginner") {
    state.view = "beginner";
    renderBeginner();
    return;
  }
  renderHome();
}

function renderHome() {
  app.innerHTML = `
    <section class="hero">
      <div>
        <p class="eyebrow">Typing Practice</p>
        <h1>打字练习</h1>
        <p class="lead">从手指位置开始，慢慢练到英文和中文文章。每次练习结束都会显示正确率。</p>
      </div>
    </section>
    <section class="category-grid" aria-label="课程入口">
      ${categoryCards
        .map(
          (card) => `
            <button class="category-card" data-action="category" data-level="${card.id}">
              <span class="card-meta">${card.meta}</span>
              <strong>${card.title}</strong>
              <span>${card.body}</span>
            </button>
          `,
        )
        .join("")}
    </section>
  `;
}

function renderBeginner() {
  app.innerHTML = `
    <header class="page-header">
      <button class="icon-text-button" data-action="home" type="button">← 首页</button>
      <div>
        <p class="eyebrow">Beginner</p>
        <h1>初阶课程</h1>
      </div>
    </header>
    <section class="lesson-grid">
      ${state.lessons
        .map(
          (lesson) => `
            <article class="lesson-card">
              <div>
                <span class="lesson-number">第 ${lesson.id} 课</span>
                <h2>${lesson.title}</h2>
                <p>${lesson.hint}</p>
              </div>
              <div class="lesson-actions">
                <button data-action="start-beginner" data-lesson="${lesson.id}" type="button">开始练习</button>
              </div>
            </article>
          `,
        )
        .join("")}
    </section>
  `;
}

async function startExercise(params) {
  const query = new URLSearchParams(params);
  const response = await fetch(`/api/exercise?${query.toString()}`);
  state.exercise = await response.json();
  state.exerciseParams = { ...params };
  state.typed = [];
  state.startedAt = Date.now();
  state.finishedAt = null;
  renderPractice();
  focusCapture();
}

function splitLines(text, lineLength) {
  const result = [];
  const chars = [...text];
  for (let i = 0; i < chars.length; i += lineLength) {
    result.push(chars.slice(i, i + lineLength).join(""));
  }
  return result;
}

function displayChar(ch) {
  if (ch === " ") return "·";
  return ch;
}

function isWideChar(ch) {
  return /[\u3000-\u303f\uff00-\uffef\u4e00-\u9fff]/u.test(ch);
}

function cellClassFor(ch) {
  return isWideChar(ch) ? " wide" : "";
}

function renderPractice() {
  const exercise = state.exercise;
  const target = exercise.text;
  const lines = splitLines(target, exercise.lineLength);
  let offset = 0;

  const rows = lines
    .map((line) => {
      const targetSpans = [...line]
        .map((ch, index) => {
          const absolute = offset + index;
          const active = absolute === state.typed.length ? " active" : "";
          const spaceClass = ch === " " ? " space" : "";
          return `<span class="cell${active}${spaceClass}${cellClassFor(ch)}">${escapeHtml(displayChar(ch))}</span>`;
        })
        .join("");
      const typedSpans = [...line]
        .map((targetChar, index) => {
          const typedChar = state.typed[offset + index];
          const widthClass = cellClassFor(targetChar);
          if (typedChar === undefined) {
            return `<span class="cell empty${widthClass}">&nbsp;</span>`;
          }
          const result = typedChar === targetChar ? "correct" : "wrong";
          const spaceClass = typedChar === " " ? " space" : "";
          return `<span class="cell ${result}${spaceClass}${widthClass}">${escapeHtml(displayChar(typedChar))}</span>`;
        })
        .join("");
      offset += line.length;
      return `
        <div class="line-pair">
          <div class="target-line">${targetSpans}</div>
          <div class="typed-line">${typedSpans}</div>
        </div>
      `;
    })
    .join("");

  app.innerHTML = `
    <header class="practice-header">
      <div>
        <p class="eyebrow">${exercise.level}</p>
        <h1>${exercise.title}</h1>
        <p>${exercise.subtitle}</p>
      </div>
      <div class="practice-actions">
        <button class="secondary" data-action="back" type="button">返回</button>
        <button data-action="restart" type="button">重新开始</button>
      </div>
    </header>
    <section class="stats-bar" aria-label="练习进度">${statsMarkup()}</section>
    <section class="practice-board level-${exercise.level}" data-action="focus">
      ${rows}
    </section>
    ${state.finishedAt ? resultPanel() : ""}
  `;
}

function statsMarkup() {
  const target = state.exercise.text;
  const speedStat = shouldShowSpeed()
    ? `<div><span>${keystrokesPerSecond()}</span><small>按键/秒</small></div>`
    : "";
  return `
    <div><span>${state.typed.length}</span><small>已输入</small></div>
    <div><span>${target.length}</span><small>总字符</small></div>
    <div><span>${currentAccuracy()}%</span><small>当前正确率</small></div>
    ${speedStat}
  `;
}

function currentAccuracy() {
  if (state.typed.length === 0) return 100;
  const target = state.exercise.text;
  const correct = state.typed.filter((ch, index) => ch === target[index]).length;
  return Math.round((correct / state.typed.length) * 100);
}

function shouldShowSpeed() {
  return state.exercise?.level !== "advanced";
}

function elapsedSeconds() {
  const end = state.finishedAt || Date.now();
  return Math.max(1, (end - state.startedAt) / 1000);
}

function keystrokesPerSecond() {
  return (state.typed.length / elapsedSeconds()).toFixed(2);
}

function resultPanel() {
  const target = state.exercise.text;
  const correct = state.typed.filter((ch, index) => ch === target[index]).length;
  const total = target.length;
  const seconds = Math.max(1, Math.round((state.finishedAt - state.startedAt) / 1000));
  const speedResult = shouldShowSpeed() ? `<span>按键/秒 ${keystrokesPerSecond()}</span>` : "";
  return `
    <aside class="result-panel" role="status">
      <strong>本次练习完成</strong>
      <span>正确率 ${Math.round((correct / total) * 100)}%</span>
      <span>正确 ${correct} / ${total}，用时 ${seconds} 秒</span>
      ${speedResult}
      <button data-action="restart" type="button">再练一次</button>
    </aside>
  `;
}

function addTypedText(text) {
  if (!state.exercise || state.finishedAt) return;
  const targetLength = state.exercise.text.length;
  for (const ch of text) {
    if (state.typed.length >= targetLength) break;
    if (ch === "\r" || ch === "\n") continue;
    state.typed.push(ch);
  }
  if (state.typed.length >= targetLength) {
    state.finishedAt = Date.now();
  }
  renderPractice();
  focusCapture();
}

function focusCapture() {
  capture.value = "";
  capture.focus({ preventScroll: true });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function backFromPractice() {
  if (state.exercise?.level === "beginner") {
    navigateOrRender("beginner", renderBeginner);
  } else {
    navigateOrRender("home", renderHome);
  }
}

function navigateOrRender(route, renderer) {
  const currentRoute = normalizeHash();
  if (currentRoute === route || (!currentRoute && route === "home")) {
    renderer();
  } else {
    setRoute(route);
  }
}

app.addEventListener("click", (event) => {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const action = target.dataset.action;
  if (action === "home") setRoute("home");
  if (action === "category") {
    const level = target.dataset.level;
    if (level === "beginner") setRoute("beginner");
    if (level === "intermediate") startExercise({ level });
    if (level === "advanced") startExercise({ level });
  }
  if (action === "start-beginner") {
    startExercise({ level: "beginner", lesson: target.dataset.lesson });
  }
  if (action === "restart") {
    if (window.confirm("确定重新开始吗？当前练习进度会清空。")) {
      startExercise(state.exerciseParams || { level: state.exercise.level });
    }
  }
  if (action === "back" && window.confirm("确定返回吗？当前练习进度不会保存。")) backFromPractice();
  if (action === "focus") focusCapture();
});

window.addEventListener("hashchange", render);

capture.addEventListener("keydown", (event) => {
  if (event.key === "Backspace" || event.key === "Delete") {
    event.preventDefault();
  }
});

capture.addEventListener("paste", (event) => {
  event.preventDefault();
});

capture.addEventListener("compositionstart", () => {
  state.composing = true;
});

capture.addEventListener("compositionend", (event) => {
  state.composing = false;
  window.setTimeout(() => {
    if (capture.value) {
      addTypedText(capture.value);
      capture.value = "";
    }
  }, 0);
});

capture.addEventListener("input", () => {
  if (state.composing) return;
  if (capture.value) {
    addTypedText(capture.value);
    capture.value = "";
  }
});

loadLessons().then(render);
