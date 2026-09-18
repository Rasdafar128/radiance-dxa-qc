"use strict";
const $ = (id) => document.getElementById(id);
let file = null,
  rows = [],
  csv = "",
  selected = 0,
  filter = "all",
  demo = false,
  busy = false;
const spine = "Поясничный отдел позвоночника";
const hip = "Проксимальный отдел бедра";
const criteria = {
  [spine]: [
    "Некорректная укладка",
    "Не выравнена ось позвоночника",
    "Присутствуют посторонние предметы",
  ],
  [hip]: ["Некорректная укладка", "Некорректная область интереса"],
};
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
const failed = (row) => row.processing_status !== "Success";
const attention = (row) => !failed(row) && Number(row.quality_class) === 1;
const basename = (name) =>
  String(name || "Без имени")
    .split(/[\\/]/)
    .pop();
const regionLabel = (row) =>
  row.anatomical_region === spine
    ? "Поясничный отдел"
    : row.anatomical_region === hip
      ? "Проксимальный отдел бедра"
      : "Область не определена";
const statusText = (row) =>
  failed(row)
    ? "Не обработан"
    : attention(row)
      ? "Есть находки"
      : "Без находок";
const statusClass = (row) =>
  failed(row) ? "failed" : attention(row) ? "warn" : "ok";
function error(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
function route() {
  const model = location.hash === "#model";
  $("check-view").hidden = model;
  $("model-view").hidden = !model;
  for (const [id, active] of [
    ["nav-check", !model],
    ["nav-model", model],
  ]) {
    $(id).classList.toggle("active", active);
    if (active) $(id).setAttribute("aria-current", "page");
    else $(id).removeAttribute("aria-current");
  }
}
window.addEventListener("hashchange", route);
route();
async function health() {
  try {
    const response = await fetch("/api/health", {
      signal: AbortSignal.timeout(7000),
    });
    if (!response.ok) throw new Error();
    const data = await response.json();
    $("connection").className = "connection online";
    $("connection-text").textContent = data.busy
      ? "Модель занята"
      : "Модель готова";
    $("connection").title = data.busy
      ? "Модель обрабатывает другой пакет"
      : "Соединение с E5 установлено";
  } catch {
    $("connection").className = "connection offline";
    $("connection-text").textContent = "Модель недоступна";
    $("connection").title =
      "Сервис модели недоступен. Пример интерфейса работает без подключения.";
  }
}
health();
function choose(candidate) {
  if (busy || !candidate) return;
  if (candidate.size > 256 * 1024 * 1024) {
    error("Файл больше 256 МиБ. Разделите его на несколько архивов.");
    return;
  }
  if (!candidate.size) {
    error("Этот файл пуст. Выберите другой файл.");
    return;
  }
  if (
    !/\.(zip|dcm|dicom)$/i.test(candidate.name) &&
    !["application/dicom", "application/zip"].includes(candidate.type)
  ) {
    error(
      "Выберите DICOM (.dcm, .dicom) или ZIP. Файл DICOM без расширения можно упаковать в ZIP.",
    );
    return;
  }
  file = candidate;
  error("");
  $("file-heading").textContent = candidate.name;
  $("file-description").textContent =
    `${(candidate.size / 1024 / 1024).toLocaleString("ru-RU", { maximumFractionDigits: 2 })} МиБ · Готов к отправке`;
  $("choose-file").hidden = true;
  $("analyze").hidden = false;
  $("clear-file").hidden = false;
  $("analyze").focus();
}
function clearFile() {
  file = null;
  $("file-input").value = "";
  error("");
  $("file-heading").textContent = "Перетащите файл сюда";
  $("file-description").textContent =
    "Один DICOM или ZIP с несколькими снимками. До 256 МиБ, без пароля.";
  $("choose-file").hidden = false;
  $("analyze").hidden = true;
  $("clear-file").hidden = true;
}
$("choose-file").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) =>
  choose(event.target.files[0]),
);
$("clear-file").addEventListener("click", () => {
  clearFile();
  $("choose-file").focus();
});
const zone = $("drop-zone");
for (const type of ["dragenter", "dragover"])
  zone.addEventListener(type, (event) => {
    event.preventDefault();
    if (!busy) zone.classList.add("drag-over");
  });
for (const type of ["dragleave", "drop"])
  zone.addEventListener(type, (event) => {
    event.preventDefault();
    zone.classList.remove("drag-over");
  });
zone.addEventListener("drop", (event) => {
  if (busy) return;
  if (event.dataTransfer.files.length !== 1) {
    error("Выберите один файл. Несколько DICOM упакуйте в ZIP.");
    return;
  }
  choose(event.dataTransfer.files[0]);
});
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
window.addEventListener("beforeunload", (event) => {
  if (busy) {
    event.preventDefault();
    event.returnValue = "";
  }
});
function setBusy(value) {
  busy = value;
  $("progress").hidden = !value;
  $("upload-workspace").setAttribute("aria-busy", String(value));
  for (const id of [
    "choose-file",
    "analyze",
    "clear-file",
    "show-demo",
    "file-input",
  ])
    $(id).disabled = value;
}
function request(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/analyze?filename=" + encodeURIComponent(file.name));
    xhr.setRequestHeader(
      "Content-Type",
      /\.zip$/i.test(file.name) || file.type === "application/zip"
        ? "application/zip"
        : "application/dicom",
    );
    xhr.timeout = 625000;
    xhr.upload.onload = () => {
      $("progress-text").textContent = "Модель проверяет снимки…";
    };
    xhr.onload = () => {
      let response;
      try {
        response = JSON.parse(xhr.responseText);
      } catch {
        reject(
          new Error(
            "Сервис вернул неожиданный ответ. Повторите проверку позже.",
          ),
        );
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(response);
      else
        reject(
          new Error(
            typeof response.detail === "string"
              ? response.detail
              : "Проверка не выполнена. Попробуйте другой файл.",
          ),
        );
    };
    xhr.onerror = () =>
      reject(
        new Error(
          "Соединение прервалось. Проверьте сеть и повторите отправку.",
        ),
      );
    xhr.ontimeout = () =>
      reject(new Error("Время ожидания истекло. Попробуйте меньший пакет."));
    xhr.send(file);
  });
}
$("analyze").addEventListener("click", async () => {
  if (!file || busy) return;
  error("");
  $("progress-text").textContent = "Передаём файл…";
  setBusy(true);
  try {
    const result = await request(file);
    if (
      !Array.isArray(result.rows) ||
      !result.rows.length ||
      typeof result.csv !== "string"
    )
      throw new Error("Ответ модели неполный. Повторите проверку.");
    rows = result.rows;
    csv = result.csv;
    demo = false;
    showResults();
  } catch (exception) {
    error(exception.message);
  } finally {
    setBusy(false);
    health();
  }
});
function showResults() {
  filter = "all";
  selected = rows.findIndex(attention);
  if (selected < 0) selected = 0;
  $("upload-workspace").hidden = true;
  $("results").hidden = false;
  $("demo-notice").hidden = !demo;
  $("result-summary").textContent =
    `${rows.length} файлов · ${rows.filter(attention).length} с находками · ${rows.filter(failed).length} не обработано${demo ? "" : " · E5"}`;
  $("count-all").textContent = rows.length;
  $("count-attention").textContent = rows.filter(attention).length;
  $("count-failure").textContent = rows.filter(failed).length;
  render();
  $("results-heading").focus({ preventScroll: true });
  $("results").scrollIntoView({ block: "start" });
}
function openResult(index) {
  selected = index;
  render();
  if (matchMedia("(max-width: 800px)").matches) {
    $("inspection-title").focus({ preventScroll: true });
    $("inspection").scrollIntoView({ block: "start" });
  } else {
    $("result-row-" + index)
      .querySelector("button")
      .focus({ preventScroll: true });
  }
}
function icon(pathData, viewBox = "0 0 24 24") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", viewBox);
  svg.setAttribute("aria-hidden", "true");
  svg.classList.add("arrow-icon");
  const path = document.createElementNS(svg.namespaceURI, "path");
  path.setAttribute("d", pathData);
  svg.append(path);
  return svg;
}
function render() {
  for (const button of document.querySelectorAll("[data-filter]"))
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.filter === filter),
    );
  const visible = rows
    .map((row, index) => ({ row, index }))
    .filter(
      ({ row }) =>
        filter === "all" ||
        (filter === "attention" ? attention(row) : failed(row)),
    )
    .sort(
      (a, b) =>
        (attention(a.row) ? 0 : failed(a.row) ? 1 : 2) -
        (attention(b.row) ? 0 : failed(b.row) ? 1 : 2),
    );
  if (!visible.some(({ index }) => index === selected))
    selected = visible.length ? visible[0].index : -1;
  const body = $("result-rows");
  body.replaceChildren();
  for (const { row, index } of visible) {
    const tr = element(
      "tr",
      undefined,
      index === selected ? "result-row selected" : "result-row",
    );
    tr.id = "result-row-" + index;
    tr.addEventListener("click", () => openResult(index));
    const name = element("td");
    const button = element("button", basename(row.path_to_study), "file-link");
    button.setAttribute(
      "aria-label",
      `Открыть результат: ${basename(row.path_to_study)}`,
    );
    button.setAttribute("aria-pressed", String(index === selected));
    button.append(element("span", regionLabel(row), "file-subtitle"));
    name.append(button);
    tr.append(name);
    const status = element("td");
    status.append(
      element("span", statusText(row), "status " + statusClass(row)),
    );
    tr.append(status);
    tr.append(
      element(
        "td",
        failed(row)
          ? "—"
          : Number(row.quality_prob).toLocaleString("ru-RU", {
              minimumFractionDigits: 3,
              maximumFractionDigits: 3,
            }),
      ),
    );
    const arrow = element("td", undefined, "row-arrow");
    arrow.append(icon("M5 12h14m-6-6 6 6-6 6"));
    tr.append(arrow);
    body.append(tr);
  }
  $("empty-filter").hidden = visible.length > 0;
  renderInspection();
}
function renderInspection() {
  const panel = $("inspection");
  panel.replaceChildren();
  const row = rows[selected];
  const title = element(
    "h2",
    row ? basename(row.path_to_study) : "Нет выбранного снимка",
  );
  title.id = "inspection-title";
  title.tabIndex = -1;
  panel.append(title);
  if (!row) {
    panel.append(
      element(
        "p",
        "Выберите другую категорию, чтобы просмотреть результаты.",
        "inspection-note",
      ),
    );
    return;
  }
  const back = element(
    "button",
    "К выбранному файлу",
    "text-button back-to-row",
  );
  back.addEventListener("click", () => {
    const tr = $("result-row-" + selected);
    tr.querySelector("button").focus({ preventScroll: true });
    tr.scrollIntoView({ block: "center" });
  });
  panel.append(back);
  panel.append(element("p", regionLabel(row), "inspection-region"));
  panel.append(element("span", statusText(row), "status " + statusClass(row)));
  if (failed(row)) {
    panel.append(
      element(
        "p",
        "Файл не удалось обработать. Проверьте, что это поддерживаемый однокадровый монохромный DICOM, и повторите отправку.",
        "inspection-note",
      ),
    );
  } else {
    panel.append(element("h3", "Критерии проверки"));
    const list = element("ul", undefined, "finding-list");
    const found = String(row.violation_type || "")
      .split("; ")
      .filter(Boolean);
    for (const label of criteria[row.anatomical_region] || []) {
      const yes = found.includes(label);
      const item = element("li", undefined, yes ? "found" : "not-found");
      const symbol = icon(
        yes ? "M10 3 18 17H2L10 3Zm0 4v5m0 2v1" : "m4 10 4 4 8-8",
        "0 0 20 20",
      );
      const copy = element("span", label, "finding-copy");
      copy.append(
        element("small", yes ? "Отмечено моделью" : "Не отмечено моделью"),
      );
      item.append(symbol, copy);
      list.append(item);
    }
    panel.append(list);
    panel.append(
      element(
        "p",
        `Оценка нарушения: ${Number(row.quality_prob).toLocaleString("ru-RU", { minimumFractionDigits: 3, maximumFractionDigits: 3 })}. ${attention(row) ? "Проверьте отмеченные критерии на исходном снимке." : "Отсутствие находок не гарантирует отсутствие нарушений."}`,
        "inspection-note",
      ),
    );
  }
  const details = element("details", undefined, "metadata");
  details.append(element("summary", "Данные файла"));
  const dl = element("dl");
  for (const [label, value] of [
    ["Путь", row.path_to_study],
    ["Study UID", row.study_uid],
    ["Image UID", row.image_uid],
    ["Время обработки, с", row.time_of_processing],
  ]) {
    dl.append(element("dt", label), element("dd", value || "—"));
  }
  details.append(dl);
  panel.append(details);
}
for (const button of document.querySelectorAll("[data-filter]"))
  button.addEventListener("click", () => {
    filter = button.dataset.filter;
    render();
  });
$("new-upload").addEventListener("click", () => {
  rows = [];
  csv = "";
  demo = false;
  $("results").hidden = true;
  $("upload-workspace").hidden = false;
  clearFile();
  $("choose-file").focus();
});
$("download").addEventListener("click", () => {
  const url = URL.createObjectURL(
    new Blob(["\ufeff", csv], { type: "text/csv;charset=utf-8" }),
  );
  const link = element("a");
  link.href = url;
  link.download = demo ? "dxa-synthetic-example.csv" : "dxa-results.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("show-demo").addEventListener("click", () => {
  const base = {
    study_uid: "SYNTHETIC-STUDY",
    image_uid: "SYNTHETIC-IMAGE",
    processing_status: "Success",
    time_of_processing: "",
  };
  rows = [
    {
      ...base,
      path_to_study: "Пример / spine_01.dcm",
      anatomical_region: spine,
      quality_class: "1",
      quality_prob: "0.782",
      violation_type: "Присутствуют посторонние предметы",
    },
    {
      ...base,
      path_to_study: "Пример / hip_01.dcm",
      anatomical_region: hip,
      quality_class: "0",
      quality_prob: "0.164",
      violation_type: "",
    },
    {
      ...base,
      path_to_study: "Пример / hip_02.dcm",
      anatomical_region: hip,
      quality_class: "1",
      quality_prob: "0.643",
      violation_type: "Некорректная укладка",
    },
    {
      ...base,
      path_to_study: "Пример / unreadable.dcm",
      anatomical_region: "",
      quality_class: "",
      quality_prob: "",
      violation_type: "",
      processing_status: "Failure",
    },
  ];
  const columns = [
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",
    "quality_prob",
    "violation_type",
    "processing_status",
    "time_of_processing",
  ];
  const quote = (value) =>
    '"' + String(value || "").replaceAll('"', '""') + '"';
  csv =
    columns.join(",") +
    "\n" +
    rows
      .map((row) => columns.map((key) => quote(row[key])).join(","))
      .join("\n");
  demo = true;
  showResults();
});
