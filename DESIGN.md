---
name: DXA Контроль
description: A clear inspection sheet for DXA acquisition-quality results.
colors:
  blue: "#264fa5"
  blue-soft: "#edf2fc"
  paper: "#fff"
  ground: "#f1f3f6"
  ink: "#192a46"
  muted: "#59677c"
  line: "#d9dfe8"
  ok: "#266b50"
  ok-bg: "#eaf4ee"
  warn: "#89550b"
  warn-bg: "#fbf1dc"
  error: "#9a3536"
  error-bg: "#fff0ef"
typography:
  display:
    fontFamily: "Golos, system-ui, sans-serif"
    fontSize: "52px"
    fontWeight: 550
    lineHeight: 1.1
    letterSpacing: "-0.035em"
  title:
    fontFamily: "Golos, system-ui, sans-serif"
    fontSize: "20px"
    fontWeight: 550
    lineHeight: 1.3
    letterSpacing: "-0.02em"
  body:
    fontFamily: "Golos, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Golos, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.55
  button:
    fontFamily: "Golos, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 500
    lineHeight: 1.4
rounded:
  status: "4px"
  filter: "5px"
  control: "6px"
  inset: "8px"
  sheet: "12px"
spacing:
  compact: "8px"
  small: "12px"
  medium: "16px"
  row: "20px"
  section: "24px"
  sheet: "28px"
  roomy: "32px"
  column: "40px"
  page: "48px"
components:
  button-primary:
    backgroundColor: "{colors.blue}"
    textColor: "{colors.paper}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    padding: "11px 18px"
  button-secondary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    padding: "11px 18px"
  filter-selected:
    backgroundColor: "{colors.blue-soft}"
    textColor: "{colors.blue}"
    rounded: "{rounded.filter}"
    padding: "8px 10px"
  navigation:
    textColor: "{colors.muted}"
  status-finding:
    backgroundColor: "{colors.warn-bg}"
    textColor: "{colors.warn}"
    typography: "{typography.label}"
    rounded: "{rounded.status}"
    padding: "5px 8px"
  sheet:
    backgroundColor: "{colors.paper}"
    rounded: "{rounded.sheet}"
---

# Design System: DXA Контроль

## Overview

**Creative North Star: "The Quality Inspection Sheet"**

Cool mineral ground, white working sheets and deep blue ink make a restrained,
normally lit workspace. Cobalt marks actions and selection; ruled rows and
open explanatory margins carry the information. This records the implemented
world in `web/static/`, rather than a separate visual comp.

**Key Characteristics:**
- Clear Russian labels and a single locally hosted Cyrillic type family.
- Flat sheets, fine rules, restrained rectangular controls and generous margins.
- Textual findings and failure states, with color as supporting information.
- Authored SVG marks and icons; no decorative medical imagery or shipping raster.

## Colors

### Primary

- **Cobalt action** (`blue`): primary buttons, links, active navigation and focus-related emphasis.
- **Pale cobalt selection** (`blue-soft`): selected rows, pressed filters and drag-over feedback.

### Neutral

- **Paper / mineral ground** (`paper`, `ground`): working sheet against the page.
- **Blue ink / slate text** (`ink`, `muted`): headings and primary copy / supporting copy.
- **Fine rule** (`line`): section divisions and sheet borders.

### Status

- **Amber finding** (`warn`, `warn-bg`): a model finding requiring review.
- **Green unflagged** (`ok`, `ok-bg`): no finding reported; not a guarantee of correctness.
- **Red file failure** (`error`, `error-bg`): unprocessed files and request errors.

**The Text Before Color Rule.** Pair every operational status with readable text;
a dot, tint or icon cannot carry the meaning alone.

## Typography

Golos Text is bundled as `web/static/fonts/GolosText.ttf`, exposed as CSS family
`Golos` with weights 400–900 and `font-display: swap`. Its SIL Open Font License
1.1 is retained in `web/static/fonts/OFL.txt`; system-ui and sans-serif are fallbacks.

The frontmatter records the desktop type roles. Display headings step down to
44px at 1100px, 40px at 800px, and 34px with 1.14 line height at 480px. Body size
becomes 14px on the smallest layout. Intro copy is 16px desktop / 13px mobile;
results headings are 24px desktop / 22px mobile. Paragraphs cap at 72ch.

**The Legible Detail Rule.** Visible labels, metadata, tables and explanatory
copy stay at least 12px. Scores use tabular numerals and Russian decimal formatting.

## Layout

- Main content: centered 1344px maximum width, 48px side padding, 56px top padding.
- Desktop: fluid working sheet plus 275px explanatory margin, separated by 40px.
  Results use a 295px inspection margin and 28px gap.
- At 1100px: 32px page padding and narrower margins; at 1600px and above, more
  top and intro spacing.
- At 800px and below: sheets and margins stack; result inspection becomes a
  tinted block below the list. Upload criteria use two columns until 480px.
- At 480px and below: 20px page padding, two-row header, single-column criteria,
  stacked definitions, and equal-width result actions. The list's numeric score
  column is hidden; the selected-file explanation retains the score.
- Long filenames and metadata wrap. Tables remain real tables with a horizontal
  overflow container where necessary.

## Elevation & Depth

There are no shadows. White against mineral ground, fine borders, selected-row
tint and the mobile inspection background establish depth. The support margin
stays open on desktop rather than becoming a stack of raised cards.

## Shapes

Sheets use the largest recorded radius, with smaller corners for drop areas,
controls, filters and status labels. Borders are 1px. The drop area uses a dashed
border. Small round status dots and outlined diamond list markers are accents,
not alternate control shapes.

## Components

- **Buttons:** semantic buttons with primary, outlined secondary and quiet
  variants. Main controls are at least 46px tall; text and filename actions are
  at least 44px. Primary hover darkens to `#1d4089`; active buttons move down 1px.
  Keyboard focus is a 3px `#517cce` outline with 4px offset. Disabled upload
  actions use 0.65 opacity and a waiting cursor.
- **Navigation:** ordinary hash links, muted at rest, cobalt underline when
  active, with `aria-current="page"`. The skip link becomes visible on focus.
- **Upload:** a real file input opened by a button, plus drag-and-drop. Empty,
  selected, drag-over, uploading, processing and error states have distinct copy.
  Progress names actual stages; it does not fabricate a percentage. The workspace
  sets `aria-busy`, progress uses `role="status"`, and errors use `role="alert"`.
- **Results:** filter buttons expose `aria-pressed`; selection uses a real
  filename button inside a clickable row. Display order is stable: findings,
  failures, then clean files. Sorting mapped row/index pairs preserves original
  records and CSV order. Filters do not change the downloaded CSV.
- **Inspection:** shows the selected filename, region, readable status and only
  its applicable criteria: three spine checks or two hip checks. Metadata is a
  native details disclosure. At 800px and below, explicit selection focuses and
  reveals the inspection heading; “К выбранному файлу” returns to the same row
  and restores its button focus.
- **Status and examples:** findings, unflagged files and processing failures
  stay separate. Synthetic data has a prominent textual notice. Connection
  status distinguishes ready, busy and unavailable.
- **Icons:** authored stroked SVG; arrows share 18px geometry and 1.7px strokes.
  Meaning remains in button/link text. Decorative SVG is hidden from assistive
  technology. The brand mark is `web/static/mark.svg`.
- **Motion:** results arrive once over 0.4s with a 6px shift and shallow clip;
  controls transition over 0.15–0.2s. Reduced-motion mode disables animations,
  transitions and smooth scrolling.

## Do's and Don'ts

- **Do** reuse the locally bundled family, status meanings and fine ruled surfaces.
- **Do** keep mobile selection and return paths explicit, with keyboard focus preserved.
- **Do** retain source-file identity and original CSV content when changing display order.
- **Don't** replace readable states with color-only badges or character-glyph icons.
- **Don't** present a clean result as a medical conclusion or synthetic rows as real output.
- **Don't** infer a broader visual approval from the scoped finish verdict; review
  evidence and current surface status are recorded in `.impeccable/surfaces/web.md`.


## Current review workspace · 22 September 2026

The approved workspace plan replaces the earlier results composition described above.
The incumbent colors, Golos typeface and image controls remain. The check-page heading
is now 28–38px; the model evidence heading retains its original typography.

- Above 1100px: study list (256px), fluid image viewer and inspection rail (292px),
  with 20px gaps. The list scrolls independently and stays available beside the image.
- At 801–1100px the study list becomes a native disclosure above viewer and inspection.
- At 800px and below: selected-file finding first, image second, detailed checks below.
  The list opens from the finding summary; a closed list is hidden to avoid duplicate controls.
- Positive criterion names avoid double negatives. Flagged criteria show a concise,
  expandable specification-based checklist, without simulated localization.
- Review marks are session-only and visually separate from model results. Print output
  includes all input rows regardless of the display filter, with full-frame previews.
- Status uses one current request state and elapsed time; no estimated completion percent.
  New exports stay disabled while the current operation is unfinished.

## Разметка и анатомические группы · 25 сентября 2026

Список результатов разделён на «Позвоночник», «Бёдра», «Область не определена».
Study UID сохраняется в метаданных. Разметка — прозрачный слой в пиксельной сетке
полного кадра с общими масштабом и смещением. Отдельный нативный переключатель
скрывает слой. Оранжевый #ffb454 связывает ориентир с замечанием модели, голубой
#69d4e8 показывает прочие ориентиры на тёмном фоне. На белом фоне легенды —
#c57406 и #16859b. Рядом с легендой сообщаются ограничения и ненайденные ориентиры.
Подсветка включений эвристическая; клинические ROI и миллиметры не отображаются.

## Упрощение рабочего экрана · 25 сентября 2026

- В карточках списка только имя, миниатюра, статус и отметка просмотра. Область задаёт заголовок группы.
- Найденные замечания видны сразу. Инструкции «Что проверить», остальные критерии и подробная легенда раскрываются по нажатию. Сообщения о ненайденных ориентирах остаются видимыми.
- Краткий результат на телефоне находится внутри просмотрщика, список при первом результате свёрнут. Переход к следующему замечанию не дублируется под снимком.
- Размер изображения и пояснение отметок просмотра перенесены в данные файла. Счётчик просмотренных появляется после первой отметки.
- Убраны повторные статусы, описание цветов из правой колонки и техническая подпись под снимком. Тема, шрифт и функции просмотра сохранены.

## Ожидание проверки · 26 сентября 2026

- Во время обработки в рабочем листе один статус, ниже имя файла и размер, затем время ожидания. Статус — 22 px, имя файла — 16 px. Выбранное имя до отправки — 18 px и переносится без обрезки.
- Кнопки выбора скрываются на время запроса. При ошибке выбранный файл сохраняется, кнопки возвращаются, фокус переходит на повторную проверку.
- Сообщение о ходе проверки получает фокус и собственный `role="status"`. Оно расположено вне `aria-busy`, чтобы изменения объявлялись сразу. У информационного блока нет рамки фокуса; интерактивные элементы сохраняют стандартный заметный фокус.
- Показываются фактический этап и прошедшее время. Процент готовности и ожидаемый срок не моделируются. Анимация учитывает reduced motion.

## Локальная проверка просмотрщика · 2026-09-26

- «Вписано» обозначает полный кадр с отступом 8 px; увеличение возвращает действие «Вписать». Снимок и разметка получают одинаковые размеры и преобразование.
- Полноэкранная кнопка находится в навигации снимков и позволяет выйти даже на файле без превью. На телефоне в полном экране остаются имя, навигация и инструменты изображения.
- Пояснения разметки раскрываются через «О разметке»; недоступные переключатели и переход к отсутствующему следующему замечанию скрыты.
- Локальные скриншоты и результаты браузерных проверок: `artifacts/site-audit/`.
