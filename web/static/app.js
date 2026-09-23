"use strict";
const $ = (id) => document.getElementById(id);
let files = [], rows = [], previews = [], sources = [], csv = "", selected = 0,
  filter = "all", demo = false, busy = false, completedAt = null;
const reviewed = new Set();
const spine = "Поясничный отдел позвоночника";
const hip = "Проксимальный отдел бедра";
const columns = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
  "quality_prob", "violation_type", "processing_status", "time_of_processing"];
const criteria = {
  [spine]: [
    ["Некорректная укладка", "Укладка", "Проверьте полноту кадра: половина тела Th12 сверху и верхние края подвздошных костей снизу."],
    ["Не выравнена ось позвоночника", "Ось позвоночника", "Оцените ось по центрам тел позвонков относительно вертикали кадра. По ТЗ допустим наклон до 5°. Изгиб позвоночника сам по себе не означает ошибку укладки."],
    ["Присутствуют посторонние предметы", "Посторонние предметы", "Осмотрите полный кадр на металлические предметы и наложения одежды. Модель отмечает тип нарушения, но не определяет его положение."],
  ],
  [hip]: [
    ["Некорректная укладка", "Укладка", "Проверьте видимость большого вертела, шейки бедра и седалищной кости. Оцените ротацию по контуру малого вертела согласно критериям ТЗ."],
    ["Некорректная область интереса", "Полнота поля", "Проверьте, что исследуемая зона полностью попала в поле сканирования. Эта оценка не подтверждает наличие или точность сохранённого контура ROI."],
  ],
};
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
const failed = (row) => row.processing_status !== "Success";
const attention = (row) => !failed(row) && Number(row.quality_class) === 1;
const basename = (name) => String(name || "Без имени").split(/[\\/]/).pop();
const isZip = (file) => /\.zip$/i.test(file.name) || file.type === "application/zip";
const regionLabel = (row) => row.anatomical_region === spine ? "Поясничный отдел"
  : row.anatomical_region === hip ? "Проксимальный отдел бедра" : "Область не определена";
const statusText = (row) => failed(row) ? "Не обработан" : attention(row) ? "Есть замечания" : "Без замечаний модели";
const statusClass = (row) => failed(row) ? "failed" : attention(row) ? "warn" : "ok";
const findings = (row) => String(row.violation_type || "").split(";").map(s => s.trim()).filter(Boolean);
const mainFinding = (row) => failed(row) ? "Файл не обработан" : attention(row)
  ? findings(row).join(" · ") : "Модель не отметила нарушений";
function error(message) {
  const target = $(rows.length ? "result-error" : "error");
  target.textContent = message;
  target.hidden = !message;
}
function route() {
  const model = location.hash === "#model";
  document.body.classList.toggle("review-mode", !model && rows.length > 0);
  $("check-view").hidden = model;
  $("model-view").hidden = !model;
  for (const [id, active] of [["nav-check", !model], ["nav-model", model]]) {
    $(id).classList.toggle("active", active);
    if (active) $(id).setAttribute("aria-current", "page");
    else $(id).removeAttribute("aria-current");
  }
}
window.addEventListener("hashchange", route);
route();
let healthSequence = 0;
function connection(text, online) {
  $("connection").className = "connection " + (online ? "online" : "offline");
  $("connection-text").textContent = text;
  $("connection").title = text;
}
async function health() {
  if (busy) return;
  const sequence = ++healthSequence;
  try {
    const response = await fetch("/api/health", { signal: AbortSignal.timeout(7000) });
    if (!response.ok) throw new Error();
    const data = await response.json();
    if (busy || sequence !== healthSequence) return;
    connection(data.busy ? "Модель занята" : "Модель готова", true);
  } catch {
    if (!busy && sequence === healthSequence) connection("Модель недоступна", false);
  }
}
health();
window.addEventListener("focus", health);
setInterval(() => { if (!document.hidden) health(); }, 30000);
function choose(candidates) {
  if (busy || !candidates.length) return;
  const next = Array.from(candidates);
  if (next.length > 1000 || next.reduce((sum, f) => sum + f.size, 0) > 256 * 1024**2) {
    error("Выберите до 1000 файлов общим размером не больше 256 МиБ."); return;
  }
  if (next.some(f => !f.size)) { error("В выборе есть пустой файл. Уберите его и повторите выбор."); return; }
  if (next.some(f => !/\.(zip|dcm|dicom)$/i.test(f.name) && !["application/dicom", "application/zip"].includes(f.type))) {
    error("Выберите DICOM (.dcm, .dicom) или ZIP. DICOM без расширения можно упаковать в ZIP."); return;
  }
  if (next.length > 1 && next.some(isZip)) { error("Выберите несколько DICOM или один ZIP отдельно."); return; }
  files = next;
  error("");
  $("file-heading").textContent = next.length === 1 ? next[0].name : `Выбрано файлов: ${next.length}`;
  $("file-description").textContent = `${(next.reduce((n, f) => n + f.size, 0) / 1024**2).toLocaleString("ru-RU", {maximumFractionDigits: 2})} МиБ · Готово к проверке`;
  $("choose-file").hidden = true;
  $("analyze").hidden = $("clear-file").hidden = false;
  $("analyze").focus();
}
function clearFiles() {
  files = [];
  $("file-input").value = "";
  error("");
  $("file-heading").textContent = "Перетащите снимки сюда";
  $("file-description").textContent = "Несколько DICOM или один ZIP с исследованиями. До 256 МиБ суммарно, ZIP без пароля.";
  $("choose-file").hidden = false;
  $("analyze").hidden = $("clear-file").hidden = true;
}
$("choose-file").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", event => choose(event.target.files));
$("clear-file").addEventListener("click", () => { clearFiles(); $("choose-file").focus(); });
const zone = $("drop-zone");
for (const type of ["dragenter", "dragover"]) zone.addEventListener(type, event => {
  event.preventDefault(); if (!busy) zone.classList.add("drag-over");
});
for (const type of ["dragleave", "drop"]) zone.addEventListener(type, event => {
  event.preventDefault(); zone.classList.remove("drag-over");
});
zone.addEventListener("drop", event => choose(event.dataTransfer.files));
window.addEventListener("dragover", event => event.preventDefault());
window.addEventListener("drop", event => event.preventDefault());
window.addEventListener("beforeunload", event => {
  if (busy || (rows.length && !demo)) { event.preventDefault(); event.returnValue = ""; }
});
let started = 0, timer;
function tick() {
  const elapsed = `Прошло ${Math.floor((Date.now() - started) / 1000)} с`;
  $("elapsed").textContent = elapsed;
  // Таймер не объявляется скринридеру каждую секунду.
  $("result-elapsed").textContent = elapsed;
}
function progress(text) {
  $("result-progress-text").textContent = text;
  $("progress-text").textContent = text;
  $("file-description").textContent = text;
  connection("Идёт проверка", true);
  tick();
}
function setBusy(value) {
  busy = value;
  ++healthSequence;
  $("progress").hidden = !value;
  $("result-progress").hidden = !value || !rows.length;
  $("upload-workspace").setAttribute("aria-busy", String(value));
  for (const id of ["choose-file", "analyze", "clear-file", "show-demo", "file-input", "new-upload", "download", "print"])
    $(id).disabled = value;
  clearInterval(timer);
  if (value) { started = Date.now(); timer = setInterval(tick, 1000); }
}
function request(file, imageIndex = null, label = "") {
  progress(`${label}Передаём файл…`);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const params = new URLSearchParams({filename: file.name});
    if (imageIndex !== null) params.set("image_index", imageIndex);
    xhr.open("POST", "/api/analyze?" + params);
    xhr.setRequestHeader("Content-Type", isZip(file) ? "application/zip" : "application/dicom");
    xhr.timeout = 625000;
    xhr.upload.onload = () => progress(`${label}Ожидаем результат модели…`);
    xhr.onload = () => {
      let result;
      try { result = JSON.parse(xhr.responseText); }
      catch { reject(new Error("Сервис вернул неожиданный ответ. Повторите проверку.")); return; }
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(typeof result.detail === "string" ? result.detail : "Проверка не выполнена.")); return;
      }
      if (!Array.isArray(result.rows) || !result.rows.length || typeof result.csv !== "string"
          || result.rows.some(r => columns.some(key => !(key in r)))) {
        reject(new Error("Ответ модели неполный. Повторите проверку.")); return;
      }
      resolve(result);
    };
    xhr.onerror = () => reject(new Error("Соединение прервалось. Проверьте сеть и повторите отправку."));
    xhr.ontimeout = () => reject(new Error("Время ожидания истекло. Попробуйте меньший пакет."));
    xhr.send(file);
  });
}
function exportRows() {
  const quote = value => '"' + String(value ?? "").replaceAll('"', '""') + '"';
  return columns.join(",") + "\n" + rows.map(row => columns.map(key => quote(row[key])).join(",")).join("\n") + "\n";
}
$("analyze").addEventListener("click", async () => {
  if (!files.length || busy) return;
  error("");
  rows = []; previews = []; sources = []; reviewed.clear(); demo = false; selected = 0; filter = "all";
  setBusy(true);
  try {
    for (const [index, file] of files.entries()) {
      try {
        const result = await request(file, null, files.length > 1 ? `Файл ${index + 1} из ${files.length} · ` : "");
        for (const [i, row] of result.rows.entries()) {
          rows.push(row); previews.push(result.previews?.[i] || {});
          sources.push({file, imageIndex: isZip(file) ? i : null});
        }
        csv = files.length === 1 ? result.csv : exportRows();
      } catch (exception) {
        if (isZip(file)) throw exception;
        rows.push({...Object.fromEntries(columns.map(c => [c, ""])), path_to_study: file.name, processing_status: "Failure"});
        previews.push({message: exception.message}); sources.push({file, imageIndex: null});
        csv = exportRows();
      }
      showResults(false);
    }
    completedAt = new Date();
  } catch (exception) { error(exception.message); }
  finally {
    setBusy(false); health();
    if (rows.length) { render(); $("results-heading").focus({preventScroll: true}); }
  }
});
function showResults(focus = true) {
  $("check-view").classList.add("has-results");
  route();
  $("upload-workspace").hidden = true;
  $("results").hidden = false;
  $("demo-notice").hidden = !demo;
  $("result-progress").hidden = !busy;
  render();
  if (focus) $("results-heading").focus({preventScroll: true});
}
function openResult(index) {
  selected = index;
  render();
  if (matchMedia("(max-width: 1100px)").matches) $("study-list").open = false;
  const target = $(matchMedia("(max-width: 800px)").matches ? "mobile-summary" : "inspection-title");
  target.focus({preventScroll: true});
  if (matchMedia("(max-width: 800px)").matches) target.scrollIntoView({block: "start"});
}
function icon(pathData, viewBox = "0 0 24 24") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", viewBox); svg.setAttribute("aria-hidden", "true"); svg.classList.add("arrow-icon");
  const path = document.createElementNS(svg.namespaceURI, "path"); path.setAttribute("d", pathData); svg.append(path); return svg;
}
const view = { zoom: 1, x: 0, y: 0, index: -1, source: "", visible: [] };
const stage = $("viewer-stage");
const scan = $("dicom-image");
let dragging = null;
function applyView() {
  const active = !scan.hidden;
  const width = scan.naturalWidth || stage.clientWidth;
  const height = scan.naturalHeight || stage.clientHeight;
  const fit = Math.min(stage.clientWidth / width, stage.clientHeight / height);
  const maxX = Math.max(0, (width * fit * view.zoom - stage.clientWidth) / 2);
  const maxY = Math.max(0, (height * fit * view.zoom - stage.clientHeight) / 2);
  view.x = Math.max(-maxX, Math.min(maxX, view.x));
  view.y = Math.max(-maxY, Math.min(maxY, view.y));
  scan.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`;
  scan.style.filter = `brightness(${$("image-brightness").value}%) contrast(${$("image-contrast").value}%)`;
  $("zoom-value").textContent = view.zoom.toLocaleString("ru-RU", { minimumFractionDigits: 1 }) + "×";
  $("brightness-value").textContent = $("image-brightness").value + "%";
  $("contrast-value").textContent = $("image-contrast").value + "%";
  $("zoom-out").disabled = !active || view.zoom <= 1;
  $("zoom-in").disabled = !active || view.zoom >= 4;
  for (const id of ["fit-image", "reset-image", "image-brightness", "image-contrast"])
    $(id).disabled = !active;
  stage.classList.toggle("is-zoomed", active && view.zoom > 1);
}
function resetView() {
  view.zoom = 1;
  view.x = view.y = 0;
  $("image-brightness").value = $("image-contrast").value = "100";
  applyView();
}
function zoomBy(delta) {
  if (scan.hidden) return;
  view.zoom = Math.max(1, Math.min(4, view.zoom + delta));
  applyView();
}
function renderViewer(visible) {
  view.visible = visible.map(({ index }) => index);
  const position = view.visible.indexOf(selected);
  $("image-position").textContent = position < 0 ? "0 / 0" : `${position + 1} / ${visible.length}`;
  $("previous-image").disabled = position <= 0;
  $("next-image").disabled = position < 0 || position === visible.length - 1;
  const row = rows[selected];
  const preview = previews[selected];
  const source = !demo && preview?.image?.startsWith("data:image/png;base64,") ? preview.image : "";
  $("viewer-title").textContent = row ? basename(row.path_to_study) : "Нет выбранного снимка";
  $("viewer-caption").textContent = row
    ? `${regionLabel(row)}${source ? ` · ${preview.width} × ${preview.height} px${preview.reduced ? " · уменьшенное превью" : ""}` : ""}`
    : "Выберите другую категорию в списке файлов.";
  scan.hidden = !source;
  $("viewer").classList.toggle("is-empty", !source);
  $("viewer-empty").hidden = Boolean(source);
  $("viewer-message").textContent = !row ? "В этой категории нет снимков."
    : demo ? "В учебном примере нет DICOM. Загрузите свой файл, чтобы рассмотреть снимок."
    : preview?.message || "Превью недоступно. Результат проверки и CSV сохранены.";
  if (view.index !== selected || view.source !== source) {
    view.index = selected;
    view.source = source;
    if (source) {
      scan.src = source;
      scan.alt = `Полный кадр DICOM: ${basename(row.path_to_study)}. ${regionLabel(row)}.`;
    } else scan.removeAttribute("src");
    resetView();
  }
  applyView();
}
function moveImage(delta) {
  const next = view.visible[view.visible.indexOf(selected) + delta];
  if (next === undefined) return;
  openResult(next);
}
$("previous-image").addEventListener("click", () => moveImage(-1));
$("next-image").addEventListener("click", () => moveImage(1));
$("zoom-in").addEventListener("click", () => zoomBy(0.5));
$("zoom-out").addEventListener("click", () => zoomBy(-0.5));
$("fit-image").addEventListener("click", () => {
  view.zoom = 1;
  view.x = view.y = 0;
  applyView();
});
$("reset-image").addEventListener("click", resetView);
for (const id of ["image-brightness", "image-contrast"])
  $(id).addEventListener("input", applyView);
stage.addEventListener("keydown", (event) => {
  if (scan.hidden) return;
  if (["+", "="].includes(event.key)) zoomBy(0.5);
  else if (event.key === "-") zoomBy(-0.5);
  else if (event.key === "0") resetView();
  else if (event.key.startsWith("Arrow") && view.zoom > 1) {
    if (event.key === "ArrowLeft") view.x += 30;
    if (event.key === "ArrowRight") view.x -= 30;
    if (event.key === "ArrowUp") view.y += 30;
    if (event.key === "ArrowDown") view.y -= 30;
    applyView();
  } else return;
  event.preventDefault();
});
stage.addEventListener("pointerdown", (event) => {
  if (scan.hidden || view.zoom <= 1 || event.button !== 0 || !event.isPrimary) return;
  dragging = { id: event.pointerId, x: event.clientX - view.x, y: event.clientY - view.y };
  stage.setPointerCapture(event.pointerId);
});
stage.addEventListener("pointermove", (event) => {
  if (dragging?.id !== event.pointerId) return;
  view.x = event.clientX - dragging.x;
  view.y = event.clientY - dragging.y;
  applyView();
});
for (const type of ["pointerup", "pointercancel", "lostpointercapture"])
  stage.addEventListener(type, () => { dragging = null; });
const fullscreen = $("fullscreen-image");
fullscreen.hidden = !document.fullscreenEnabled;
fullscreen.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $("viewer").requestFullscreen();
  } catch {
    fullscreen.hidden = true;
  }
});
document.addEventListener("fullscreenchange", () => {
  fullscreen.setAttribute("aria-label", document.fullscreenElement ? "Выйти из полного экрана" : "Открыть просмотр на весь экран");
  fullscreen.title = document.fullscreenElement ? "Выйти из полного экрана" : "На весь экран";
  applyView();
});
new ResizeObserver(applyView).observe(stage);
scan.addEventListener("load", applyView);
scan.addEventListener("error", () => {
  scan.hidden = true;
  $("viewer").classList.add("is-empty");
  $("viewer-empty").hidden = false;
  $("viewer-message").textContent = "Не удалось показать снимок. Повторите загрузку DICOM.";
  applyView();
});
function render() {
  $("print-report").replaceChildren();
  for (const button of document.querySelectorAll("[data-filter]")) button.setAttribute("aria-pressed", String(button.dataset.filter === filter));
  const studyNumbers = new Map();
  for (const row of rows) if (row.study_uid && !studyNumbers.has(row.study_uid)) studyNumbers.set(row.study_uid, studyNumbers.size + 1);
  const visible = rows.map((row, index) => ({row, index})).filter(({row}) =>
    filter === "all" || (filter === "attention" ? attention(row) : failed(row)));
  if (!visible.some(({index}) => index === selected)) selected = visible.length ? visible[0].index : -1;
  const groups = new Map();
  for (const entry of visible) {
    const key = entry.row.study_uid || "";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(entry);
  }
  const body = $("result-rows");
  const previousScroll = body.scrollTop;
  body.replaceChildren();
  for (const [key, group] of groups) {
    const section = element("section", undefined, "study-group");
    const heading = element("h3", key ? `Исследование ${studyNumbers.get(key)}` : "Без идентификатора исследования");
    heading.append(element("span", String(group.length), "group-count"));
    section.append(heading);
    for (const {row, index} of group) {
      const button = element("button", undefined, "study-file" + (selected === index ? " selected" : ""));
      button.id = "result-row-" + index;
      button.setAttribute("aria-label", `Открыть результат: ${basename(row.path_to_study)}`);
      button.setAttribute("aria-pressed", String(selected === index));
      button.title = row.path_to_study;
      const thumb = element("span", undefined, "thumbnail");
      if (!demo && previews[index]?.image?.startsWith("data:image/png;base64,")) {
        const img = element("img"); img.src = previews[index].image; img.alt = ""; img.loading = "lazy"; thumb.append(img);
      } else thumb.append(icon("M6 3h8l4 4v14H6zM14 3v5h4"));
      const copy = element("span", undefined, "study-file-copy");
      copy.append(element("strong", basename(row.path_to_study)), element("span", regionLabel(row), "file-region"),
        element("span", statusText(row), "file-state " + statusClass(row)));
      if (reviewed.has(index)) copy.append(element("span", "Просмотрено", "reviewed-label"));
      button.append(thumb, copy); button.addEventListener("click", () => openResult(index)); section.append(button);
    }
    body.append(section);
  }
  body.scrollTop = previousScroll;
  $("empty-filter").hidden = visible.length > 0;
  $("result-summary").textContent = `${studyNumbers.size ? `Исследований: ${studyNumbers.size} · ` : ""}Файлов: ${rows.length} · С замечаниями: ${rows.filter(attention).length} · Ошибок: ${rows.filter(failed).length}`;
  $("review-summary").textContent = `Просмотрено ${reviewed.size} из ${rows.filter(r => !failed(r)).length} обработанных снимков${busy ? " · Проверка продолжается" : ""}`;
  $("count-all").textContent = rows.length;
  $("count-attention").textContent = rows.filter(attention).length;
  $("count-failure").textContent = rows.filter(failed).length;
  $("study-list-summary").textContent = `Снимки · ${rows.length}`;
  renderViewer(visible); renderInspection(); renderMobileSummary();
}
function nextAttention() {
  const order = [...rows.keys()].slice(selected + 1).concat([...rows.keys()].slice(0, selected + 1));
  const next = order.find(i => attention(rows[i]) && !reviewed.has(i) && i !== selected);
  if (next !== undefined) { filter = "all"; openResult(next); }
}
function nextButton(id) {
  const button = element("button", "Следующее с замечанием", "button secondary next-attention");
  button.id = id;
  button.disabled = !rows.some((r, i) => i !== selected && attention(r) && !reviewed.has(i));
  button.addEventListener("click", nextAttention); return button;
}
function renderMobileSummary() {
  const panel = $("mobile-summary"); panel.replaceChildren();
  const row = rows[selected]; if (!row) return;
  panel.append(element("p", basename(row.path_to_study) + " · " + regionLabel(row), "mobile-file"),
    element("h3", mainFinding(row), "mobile-finding " + statusClass(row)));
  const nav = element("div", undefined, "mobile-navigation");
  const list = element("button", "Список снимков", "text-button");
  list.addEventListener("click", () => {
    $("study-list").open = true;
    $("study-list-summary").focus({preventScroll: true}); $("study-list").scrollIntoView({block: "start"});
  });
  nav.append(list, nextButton("next-attention-mobile")); panel.append(nav);
}
function renderInspection() {
  const panel = $("inspection"); panel.replaceChildren();
  const row = rows[selected];
  const title = element("h2", row ? (failed(row) ? "Не удалось проверить" : "Результат проверки") : "Нет выбранного снимка");
  title.id = "inspection-title"; title.tabIndex = -1; panel.append(title);
  if (!row) { panel.append(element("p", "Выберите другую категорию снимков.", "inspection-note")); return; }
  panel.append(element("p", regionLabel(row), "inspection-region"), element("span", statusText(row), "status " + statusClass(row)));
  if (failed(row)) {
    panel.append(element("p", previews[selected]?.message || "Проверьте, что файл — поддерживаемый однокадровый монохромный DICOM.", "inspection-note"));
    if (!demo && sources[selected]) {
      const retry = element("button", busy ? "Идёт проверка…" : "Повторить этот файл", "button secondary");
      retry.id = "retry-file"; retry.disabled = busy;
      retry.addEventListener("click", () => retryFile(selected)); panel.append(retry);
      panel.append(element("p", "Повторится только этот файл. Успешные результаты останутся. Если файл повреждён, загрузите исправную копию новой проверкой.", "inspection-note"));
    }
  } else {
    const found = findings(row);
    const list = element("ul", undefined, "finding-list");
    const checks = [...(criteria[row.anatomical_region] || [])].sort((a, b) => Number(found.includes(b[0])) - Number(found.includes(a[0])));
    for (const [key, label, help] of checks) {
      const yes = found.includes(key);
      const item = element("li", undefined, yes ? "found" : "not-found");
      item.append(icon(yes ? "M10 3 18 17H2L10 3Zm0 4v5m0 2v1" : "m4 10 4 4 8-8", "0 0 20 20"));
      const copy = element("div", undefined, "finding-copy");
      copy.append(element("strong", label), element("small", yes ? "Модель отметила нарушение" : "Не отмечено моделью"));
      if (yes) {
        const explanation = element("details", undefined, "criterion-help");
        explanation.open = true;
        explanation.append(element("summary", "Что проверить"), element("p", help)); copy.append(explanation);
      }
      item.append(copy); list.append(item);
    }
    panel.append(list);
    const mark = element("button", reviewed.has(selected) ? "Просмотрено · отменить" : "Отметить просмотренным", "button " + (reviewed.has(selected) ? "secondary" : "primary"));
    mark.id = "mark-reviewed"; mark.disabled = busy; mark.setAttribute("aria-pressed", String(reviewed.has(selected)));
    mark.addEventListener("click", () => {
      if (reviewed.has(selected)) reviewed.delete(selected); else reviewed.add(selected);
      render(); $("mark-reviewed").focus({preventScroll: true});
    });
    panel.append(mark, nextButton("next-attention"));
    panel.append(element("p", "Отметка означает только просмотр. Прогноз модели и CSV не меняются. Окончательное решение принимает специалист.", "inspection-note"));
  }
  const details = element("details", undefined, "metadata"); details.append(element("summary", "Данные файла и оценка"));
  const dl = element("dl");
  for (const [label, value] of [["Путь", row.path_to_study], ["Study UID", row.study_uid], ["Image UID", row.image_uid],
    ["Время обработки, с", row.time_of_processing], ["Оценка нарушения, 0–1", failed(row) ? "" : row.quality_prob], ["Модель", "Radiance 1.0"]])
    dl.append(element("dt", label), element("dd", String(value ?? "") || "—"));
  details.append(dl, element("p", "Оценка модели не является вероятностью заболевания. Отсутствие замечаний не гарантирует отсутствие нарушений."));
  panel.append(details);
}
async function retryFile(index) {
  if (busy || !sources[index]) return;
  error(""); setBusy(true); renderInspection();
  try {
    const {file, imageIndex} = sources[index];
    const result = await request(file, imageIndex, "Повтор файла · ");
    if (result.rows.length !== 1) throw new Error("Ожидался результат одного файла. Остальные результаты сохранены.");
    rows[index] = result.rows[0]; previews[index] = result.previews?.[0] || {};
    reviewed.delete(index); csv = exportRows(); completedAt = new Date();
    if (filter === "failure" && selected === index && !failed(rows[index])) filter = "all";
  } catch (exception) { error(exception.message); }
  finally { setBusy(false); health(); render(); $("inspection-title").focus({preventScroll: true}); }
}
for (const button of document.querySelectorAll("[data-filter]")) button.addEventListener("click", () => { filter = button.dataset.filter; render(); });
const compact = matchMedia("(max-width: 1100px)");
function listLayout() { $("study-list").open = !compact.matches; }
compact.addEventListener("change", listLayout); listLayout();
$("new-upload").addEventListener("click", () => {
  if (busy) return;
  $("check-view").classList.remove("has-results"); document.body.classList.remove("review-mode");
  rows = []; previews = []; sources = []; reviewed.clear(); selected = 0; csv = ""; demo = false; completedAt = null;
  view.index = -1; view.source = ""; $("dicom-image").removeAttribute("src");
  $("print-report").replaceChildren(); $("results").hidden = true; $("upload-workspace").hidden = false;
  $("result-error").hidden = true; clearFiles(); $("choose-file").focus();
});
$("download").addEventListener("click", () => {
  if (busy) return;
  const url = URL.createObjectURL(new Blob(["\ufeff", csv], {type: "text/csv;charset=utf-8"}));
  const link = element("a"); link.href = url; link.download = demo ? "dxa-synthetic-example.csv" : "dxa-results.csv";
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
function formatNumber(value, digits) {
  return value !== "" && Number.isFinite(Number(value)) ? Number(value).toLocaleString("ru-RU", {maximumFractionDigits: digits}) : "";
}
function printReport() {
  const report = $("print-report"); report.replaceChildren();
  if (!rows.length) return;
  report.append(element("h1", "Radiance · Контроль качества DXA"),
    element("p", `${demo ? "Учебный пример, вымышленные записи · " : ""}Версия 1.0 · ${(completedAt || new Date()).toLocaleString("ru-RU")}`),
    element("p", $("result-summary").textContent));
  report.append(element("p", "Результаты модели требуют проверки специалистом. Это не медицинское заключение. Отметка просмотра не подтверждает правильность прогноза."));
  for (const [index, row] of rows.entries()) {
    const section = element("article", undefined, "report-image");
    section.append(element("h2", `${index + 1}. ${basename(row.path_to_study)}`));
    const layout = element("div", undefined, "report-layout");
    if (!demo && previews[index]?.image?.startsWith("data:image/png;base64,")) {
      const img = element("img"); img.src = previews[index].image; img.alt = "Полный кадр DICOM"; layout.append(img);
    }
    const text = element("div");
    text.append(element("h3", mainFinding(row)), element("p", regionLabel(row)));
    const dl = element("dl");
    for (const [label, value] of [["Путь", row.path_to_study], ["Study UID", row.study_uid], ["Image UID", row.image_uid],
      ["Результат", statusText(row)], ["Время, с", formatNumber(row.time_of_processing, 2)], ["Оценка нарушения, 0–1", formatNumber(row.quality_prob, 3)],
      ["Отметка просмотра", reviewed.has(index) ? "Отмечен в этой сессии" : "Не отмечен"]])
      dl.append(element("dt", label), element("dd", String(value ?? "") || "—"));
    text.append(dl); layout.append(text); section.append(layout); report.append(section);
  }
}
window.addEventListener("beforeprint", () => { if (!$("print-report").children.length) printReport(); });
$("print").addEventListener("click", async () => {
  if (busy || !rows.length) return;
  printReport();
  await Promise.all([...$("print-report").querySelectorAll("img")].map(img => img.decode().catch(() => {})));
  window.print();
});
$("show-demo").addEventListener("click", () => {
  if (busy) return;
  previews = []; sources = []; reviewed.clear(); files = []; filter = "all"; selected = 0;
  const base = {...Object.fromEntries(columns.map(c => [c, ""])), study_uid: "SYNTHETIC-STUDY", processing_status: "Success"};
  rows = [
    {...base, image_uid: "SYNTHETIC-1", path_to_study: "Пример / spine_01.dcm", anatomical_region: spine, quality_class: "1", quality_prob: "0.782", violation_type: "Присутствуют посторонние предметы"},
    {...base, image_uid: "SYNTHETIC-2", path_to_study: "Пример / hip_01.dcm", anatomical_region: hip, quality_class: "0", quality_prob: "0.164"},
    {...base, image_uid: "SYNTHETIC-3", path_to_study: "Пример / hip_02.dcm", anatomical_region: hip, quality_class: "1", quality_prob: "0.643", violation_type: "Некорректная укладка"},
    {...base, study_uid: "", path_to_study: "Пример / unreadable.dcm", processing_status: "Failure"},
  ];
  csv = exportRows(); demo = true; completedAt = new Date(); showResults();
});
