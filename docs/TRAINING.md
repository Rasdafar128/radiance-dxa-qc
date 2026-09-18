# Обучение и воспроизведение

Итоговый E5 использует уже обученные головы и замороженные энкодеры.
Для обычного запуска **обучение не требуется**. Ниже — воспроизведение ML-процедуры
для разработчика с разрешённым доступом к данным организаторов и исходным весам.

## Данные и среда

Проверенная среда: Linux, Python 3.12.3, torch 2.8.0+cu126, torchvision 0.23.0+cu126,
RTX 3080 20 ГБ. Зависимости: `requirements-train.txt` и выбранная пара PyTorch.

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -r requirements-train.txt
. .venv/bin/activate
python -m src.utils.data
python -m src.utils.audit
```

Команды `python` ниже выполняются в активированной `.venv`.
В `data/raw/` нужны `НД_для_обучения.zip` и `Для теста.zip`.
Аудит сохраняет манифест, разбиения и исключения в `artifacts/audit/`.
Данные не включены в git и не скачиваются автоматически.

Политика `criteria_v2`: цели из отдельных экспертных критериев, а не противоречивых
итоговых колонок. 252 уникальных тренировочных изображения, три с неизвестными
целями исключены из supervised-обучения; остаются 249 / 100 исследований.
Протез сам по себе не является причиной исключения или положительной меткой.
Дедупликация обучения не применяется к строкам ответа при инференсе.

## Исходные веса

```bash
hf auth login
python -m src.utils.fetch dinov3-large --revision ea8dc2863c51be0a264bab82070e3e8836b02d51
python -m src.utils.fetch medimageinsight
```

Для DINOv3 учётная запись должна иметь доступ к официальному gated-репозиторию.
Токен хранится стандартными средствами HF вне проекта. MedImageInsight загружается
из официального Azure blob по фиксированным хешам. Источники, ревизии и лицензии
фиксируются в `artifacts/pretrained/*/source.json`.

## Итоговый рецепт

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m src.utils.train \
  --device cuda --backbone dinov3-large --weights artifacts/pretrained/dinov3-large \
  --quality-gate --output artifacts/reproduced-dino
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m src.utils.train \
  --device cuda --backbone medimageinsight --weights artifacts/pretrained/medimageinsight \
  --quality-gate --output artifacts/reproduced-mii
python -m src.utils.train_blend artifacts/reproduced-dino artifacts/reproduced-mii \
  --output artifacts/reproduced-e5
python -m src.utils.check_protocol artifacts/reproduced-dino artifacts/reproduced-mii artifacts/reproduced-e5
python -m src.utils.check --device cuda --model artifacts/reproduced-e5/final
```

Три внешних разбиения по исследованиям. Внутри outer train — два групповых фолда,
seed 42. Для каждой головы StandardScaler + L2 LogisticRegression/liblinear:
C ∈ {0,01; 0,1; 1}, балансировка None/balanced. Выбор типов — F1 и пороги 0,05…0,95;
головы качества выбираются по AUC, затем quality gate настраивается по внутренним
прогнозам. Область: фиксированные C=0,1, balanced=True, порог 0,5.
Ансамбль усредняет вероятности 0,5/0,5 и настраивает пороги на средних внутренних OOF.

Данные внешней проверки не участвуют в выборе голов, преобразований или порогов.
Финальные головы refit обучаются на всех 249 снимках после выбора по внутренним OOF.
Второе разбиение задаётся `--outer-seed 137` отдельно обоим родителям, затем собирается
ансамбль из этих родителей. Seed не подбирается по полученным результатам.

`recipe.json` фиксирует исходники, версии, манифест и параметры. Старые каталоги
опытов нельзя продолжать изменённым кодом: несовпадение рецепта останавливает запуск.
Используйте новые `--output`. Исторические исходники лежат в `artifacts/<опыт>/source/`.
Повторное обучение воспроизводит процедуру; побайтовое совпадение файла весов
между платформами/сериализациями не обещается. Поставка использует исходный E5
с проверенными контрольными суммами, а не автоматически заменяет его новым refit.

## Проверки и исследовательские варианты

| Команда | Назначение |
|---|---|
| `python -m src.utils.check_protocol <каталоги>` | происхождение OOF, разбиения, пороги, хеши |
| `python -m src.utils.compare <первый> <второй>` | парные интервалы различий по исследованиям |
| `python -m src.utils.diagnose <каталог>` | ошибки, редкие классы и устойчивость |
| `python -m src.utils.check_cycle3` | регрессия настройки, препроцессора, masked loss; нужны исходные артефакты DINOv3 |
| `python -m src.utils.finetune --output <новый-каталог>` | ограниченная адаптация DINOv3-L, nested CV; в поставку E5 не входит |

Исходный `e4-dinov3` содержит недействительные метрики из-за перестановки голов
при сборке внутренних прогнозов. Исправленный `e4-dinov3-v2` использует те же
обученные веса и правильную именованную сборку; старый опыт помечен `invalid.json`.
Он не может участвовать в сравнениях или новых ансамблях. Адаптированные OOF-признаки
нельзя импортировать как общий frozen-кэш.

Полные гипотезы, отрицательные результаты и причины выбора E5:
[EXPERIMENTS](../research/EXPERIMENTS.md), [ML_CYCLE3](../research/ML_CYCLE3.md).
Исследовательские веса, прогнозы и чекпойнты сохранены локально в `artifacts/`;
в публичную поставку включаются только финальные веса, агрегированные метрики и код.
