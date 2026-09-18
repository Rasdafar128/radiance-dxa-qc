---
version: 1
slug: "web"
primary_target: "web"
related_targets: []
---

# Website · quality review workspace

Operate. Upload, inspect findings, export the original CSV. Product scope and
implementation are confirmed in PRODUCT.md. The owner explicitly delegated
visual selection and chose code-first; no extra concept approval is requested.

Grounded candidates, ordered by resonance: 1 radiology light-table index;
2 laboratory accession register; 3 equipment calibration console; 4 quality
inspection sheet with a wide working column and a sign-off margin; 5 imaging
conference caseboard; 6 measurement instrument front panel; 7 clinical methods
publication. These span spatial, documentary and instrument systems. Avoid the
usual metric-card dashboard and its dark glowing scan-viewer opposite.

## Direction contract

THESIS: A quality inspection sheet: upload occupies the working area, findings
become rows to inspect, and a narrow margin explains the criteria. Every screen
belongs to a real review task.

OWN-WORLD: Cool mineral ground, near-white sheet, deep blue ink, cobalt action;
Golos Text, restrained rectangular controls, open ruled rows and generous
working margins. In a normally lit workspace, a light sheet stays legible.

STORY: Select files, start a real check, scan the findings, inspect one image's
result and download CSV. A separately labeled synthetic example teaches the
interface before upload. Model limitations have a permanent evidence view.

FIRST VIEWPORT: Compact brand/navigation band; a two-line 52px page heading;
below, a large upload surface at left and a narrow criteria rail at right.
The choose-file action sits inside the upload surface. Results reuse this
same sheet, with a row list and inspection margin. No decorative scan image.

FORM: Grounded candidate 4, assigned by seed f62eca4b. Restrained color strategy.
Signature interaction: selecting a result row transfers its applicable checks
(three for spine, two for hip) into the inspection margin, preserving row context. A single short
reveal accompanies the transition from upload to results; reduced motion is
respected. Progress names actual stages and never invents completion percent.

FINISH: built and documented in root DESIGN.md, with token extensions in
.impeccable/design.json. The finish fix pass returned `ship` for its four scored
findings; this is not a whole-surface approval. No shipping raster or approved
visual comp is part of this code-first build.

## Challenger verdicts and raises

- ASCII scene: declined on both audience identification and clarity. Raise:
  keep all informative states available as readable text, never color alone.
- Split-flap board: competitive on clarity, weaker medical identification.
  Keep stable table columns and prioritize findings before successful rows.
- Tensegrity: declined on both axes. Raise: distinguish input, connection and
  per-file failure visibly, instead of flattening all failures into one state.
- Cyclorama: declined on both axes. Raise: make processing-to-results a single
  orchestrated state change, with a reduced-motion equivalent.
- Lexicon: competitive on audience identification, weaker operational clarity.
  Keep the side margin for supporting explanation, not nested cards.
- CD-ROM console: declined on both axes. Raise: button focus and pressed states
  must be unmistakable; no faux physical texture or sound.

The ui-ux-pro-max search offered teal healthcare marketing patterns. Accepted
its contrast, touch-target and focus guidance; rejected sales funnels and
medical trust badges as unsupported by this research prototype.

## Implemented state and review evidence

- Sources: `web/static/index.html`, `style.css`, `app.js`, and `web/app.py`.
  Golos Text is local; its SIL OFL 1.1 is bundled alongside the font.
- Desktop sheet/margin layout stacks at 800px; the 480px layout uses 20px
  gutters and a two-row header. All visible text stays at least 12px.
- Rows display findings, failures, then clean results in stable order. The
  original records and exported CSV order remain unchanged. A filename button,
  status-cell click or SVG-arrow click selects the same record.
- Mobile selection reveals and focuses the inspection heading; the return
  button reveals the selected row and restores its filename-button focus.
- Upload, selected file, actual progress stages, validation/request errors,
  empty filters, file failures and ready/busy/offline connection states are
  implemented. Synthetic results remain labeled; reduced motion is respected.
- Brand and interface icons are authored SVG. Review screenshots are evidence,
  not assets served by the product.

Final captures in `.impeccable/review/`:

| View | 1440px | 375px |
| --- | --- | --- |
| Upload | `desktop.png` | `mobile.png` |
| Results | `desktop-results.png` | `mobile-results.png` |
| Model evidence | `desktop-model.png` | `mobile-model.png` |

The verdict in `.impeccable/review/verdict.md` records all four prior findings
resolved: mobile detail reveal/return, active row targets, finding priority,
and consistent SVG arrows. It reports no regressions attributable to that fix
batch; no broader approval or clinical validation is implied.
