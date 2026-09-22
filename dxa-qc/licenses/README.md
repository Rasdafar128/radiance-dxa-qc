# Лицензии сторонних весов

Сервис использует четыре замороженных энкодера. Собственный код — MIT (файл `LICENSE` в корне),
условия их авторов сохраняются отдельно:

| Энкодер | Источник | Лицензия |
|---|---|---|
| DINOv2-B (`vit_base_patch14_dinov2.lvd142m`) | timm / Meta AI | Apache 2.0 |
| RAD-DINO (`microsoft/rad-dino`) | Hugging Face / Microsoft | MSRLA (research use) |
| DINOv3-L (`dinov3-vitl16-pretrain-lvd1689m`) | Meta AI | `DINOv3.md` |
| MedImageInsight (DaViT) | Microsoft | `MedImageInsight.txt` (MIT) |

Веса энкодеров в репозиторий не кладутся: публичные скачиваются `scripts/download_weights.py`,
DINOv3-L и MedImageInsight берутся из релиза команды и раскладываются
`scripts/adapt_release_weights.py`.
