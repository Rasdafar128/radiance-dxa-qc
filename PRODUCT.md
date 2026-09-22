# DXA Контроль

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Delegated by the owner: semantic HTML, CSS and JavaScript, with a small FastAPI
gateway. The interface needs no frontend framework or GPU dependencies.

## Users

Hackathon participants and specialists checking the quality of DXA studies.
They upload DICOM files or a ZIP, inspect per-image findings, and export CSV.

## Product Purpose

Make the finished LCT 2026 quality-control pipeline usable through a browser.
Success means that a real upload produces understandable results and the exact
CSV contract, with clear handling of failures and model limitations.

## Positioning

The fixed Radiance ensemble combines DINOv3 Large and MedImageInsight to identify the
anatomical region and five acquisition-quality violations. Image-space geometry
from dxa-qc supplies the spine-axis and hip-field heads in version 1.0. This is a research
prototype; it does not diagnose osteoporosis or replace expert review.

## Operating Context

The website is hosted at https://sefixnep.ru and calls a separately hosted GPU service.
Deployment is managed with the owner. The current delivery includes the
working website, containers, documentation, model release and research audit.

## Capabilities and Constraints

- Input: supported single-frame monochrome DICOM, individually or in ZIP.
- Output: one result per file, including duplicates and failures; CSV download.
- Inspect the uploaded full-frame DICOM beside findings, with zoom, pan,
  brightness, contrast and fullscreen. Display adjustments never change inference.
- No violation masks, coordinates or heatmaps; synthetic examples contain no DICOM.
- No accounts or persistent study archive in this version.
- Files are processed temporarily; no browser-to-GPU credentials.
- The model reports image acquisition quality, not patient health.
- Primary evaluation: 249 labeled images, 100 studies; study-grouped nested CV.
- Development OOF is not an independent test; patient independence is unknown.
- Own code is MIT; pretrained component licenses remain separate.

## Brand Commitments

Working product name approved: «DXA Контроль». Russian interface, plain and
precise language. Owner delegated visual direction and implementation choices.

## Evidence on Hand

Frozen model metadata in `models/radiance/`; full local artifacts in `artifacts/`;
research reports and checked aggregate metrics in `research/`. Real DICOM data
stays outside Git. Any interface example must be clearly labeled synthetic.

## Product Principles

- Put upload and inspection ahead of marketing.
- Distinguish model findings, file errors and demonstration data.
- Show uncertainty and evaluation scope beside evidence.
- Make the exported result traceable to the original file.
