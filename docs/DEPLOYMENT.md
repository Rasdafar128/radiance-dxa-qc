# Развёртывание Radiance

## Среда

Целевая платформа контейнера — Linux x86_64, Python 3.12, PyTorch 2.8.0,
Torchvision 0.23.0. Версии остальных runtime-зависимостей закреплены в
`docker/requirements.txt`, CUDA-библиотеки — в `docker/requirements-cuda.txt`;
базовый образ закреплён digest в Dockerfile.

| Ресурс | Требование / проверка |
|---|---|
| RAM | ориентир 8 ГБ для одного процесса; измерения поставки в [DELIVERY](DELIVERY.md) |
| Диск | веса 2,48 GiB; оставьте 20 ГБ под CPU-сборку, 30 ГБ под CUDA и кэш |
| CPU | x86_64; по умолчанию 2 вычислительных потока |
| GPU | необязательна; проверена RTX 3080 20 ГБ, драйвер 580.126.09 |
| CUDA-вариант | PyTorch cu126; NVIDIA Container Toolkit на хосте |
| Сеть | нужна при сборке для зависимостей/весов; готовый инференс автономен |
| Хостовый Python | ≥3.11, только stdlib для подготовки весов |

CUDA 12.6-вариант рассчитан на проверенную RTX 3080; поддержку RTX 50xx/Blackwell
этой сборкой не заявляем. На Mac x86_64-образ работает через эмуляцию;
его скорость не является скоростью Linux-сервера.

## Веса

Точные головы и конфигурации хранятся в `models/radiance/`, SHA256 — в `selection.json`
и `weights.json`. Энкодеры прикреплены к
[GitHub Release v1.0.0](https://github.com/sefixnep/LCT26/releases/tag/v1.0.0).

**Одного клона репозитория или архива Source code (zip) недостаточно:** перед
сборкой Docker скачайте оба файла из раздела **Assets** указанного релиза.
Репозиторий приватный, поэтому для браузера нужна учётная запись с доступом.

| Файл в Release → Assets | Путь после скачивания, от корня репозитория |
|---|---|
| `dinov3-large-encoder.pt` | `models/radiance/member_0/encoder.pt` |
| `medimageinsight-encoder.pt` | `models/radiance/member_1/encoder.pt` |

`radiance-metadata.zip` содержит копию метаданных комплекта; при клонировании
репозитория распаковывать его не нужно. Лицензии и `SHA256SUMS.txt` также
приложены к релизу; проверка ниже сверяет файлы с хешами в репозитории.

Альтернатива — GitHub CLI с учётной записью, имеющей доступ:

```bash
mkdir -p artifacts/release-download models/radiance/member_0 models/radiance/member_1
gh release download v1.0.0 --repo sefixnep/LCT26 \
  --pattern '*encoder.pt' --dir artifacts/release-download
mv artifacts/release-download/dinov3-large-encoder.pt models/radiance/member_0/encoder.pt
mv artifacts/release-download/medimageinsight-encoder.pt models/radiance/member_1/encoder.pt
python3 -m src.utils.prepare_model --verify-only
```

Для сохранённого исследовательского комплекта:

```bash
python3 -m src.utils.prepare_model --from-directory artifacts/ml-final
```

Каталог источника содержит `member_0/encoder.pt` и `member_1/encoder.pt`.
Другие претрейны и повторно сериализованные файлы с иными хешами не принимаются.
Подготовка не запускает обучение. Если release assets доступны без авторизации:

```bash
python3 -m src.utils.prepare_model
```

Приватные GitHub-ссылки без авторизации отвечают 404; это не отсутствие релиза.
HF-токен не нужен. Отдельно в релиз приложены метаданные Radiance, лицензии и SHA256.

Скачивание атомарное: неполный файл не становится весами. Повреждённый уже
существующий файл вызывает ошибку; удалите указанный файл и повторите подготовку.
В runtime загрузчик повторно проверяет хеш каждого энкодера. Сборка проверяет
также головы/метаданные по `selection.json`.

## Сборка и запуск

### Сайт и модель вместе

Из корня репозитория после подготовки весов; Docker Compose ≥2.30:

```bash
docker compose -f docker/compose.yaml up -d --build --wait --wait-timeout 300
curl --fail http://127.0.0.1:8000/api/health
```

Сайт: [localhost:8000](http://127.0.0.1:8000). API: `127.0.0.1:8080`.
Compose запускает сайт после готовности модели. Оба процесса работают от
UID 10001, с read-only файловой системой и временным `/tmp`.
После перезапуска Docker сервисы поднимаются автоматически.

CUDA на Linux с NVIDIA Container Toolkit:

```bash
docker compose -f docker/compose.yaml -f docker/compose.cuda.yaml up -d --build --wait --wait-timeout 300
```

Состояние и остановка (для CUDA используйте оба `-f`):

```bash
docker compose -f docker/compose.yaml ps
docker compose -f docker/compose.yaml logs --tail 50
docker compose -f docker/compose.yaml down
```

Для обновления: получить нужный Git-тег, скачать его веса, выполнить
`prepare_model --verify-only`, затем повторить `up -d --build --wait`.
Для отката тем же способом собрать предыдущий тег; формат CSV остаётся прежним.
Локальные данные и веса при `down` не удаляются.

### Только API

```bash
./docker/run.sh cpu
# Или на NVIDIA Linux:
./docker/run.sh cuda
```

Скрипт проверяет веса, собирает образ `radiance:cpu` / `cuda`, затем запускает
API от UID 10001 с файловой системой только для чтения и временным `/tmp`.
Порт доступен только на `127.0.0.1:8080`. Один worker: каждый процесс загрузил бы
отдельную копию двух энкодеров. Холодный старт занимает десятки секунд;
готовность показывает `/health`, Docker healthcheck имеет начальный запас 120 с.

Собрать без запуска:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile -t radiance:cpu .
docker build --platform linux/amd64 --build-arg TORCH_INDEX=cu126 \
  -f docker/Dockerfile -t radiance:cuda .
```

В образ входят `src/solution`, общая конфигурация, модель, лицензии и зависимости.
Данные, разметка, HF-токен, история git и инструменты обучения исключены через
разрешающий список `.dockerignore`. Сборка использует stdlib-проверку комплекта;
её исходник удаляется из рабочего слоя после проверки. Инференс не импортирует utils.

## Запуск Python

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r docker/requirements.txt
HF_HUB_OFFLINE=1 .venv/bin/python -m src.solution.batch /path/to/input \
  --output results.csv --device cpu
HF_HUB_OFFLINE=1 .venv/bin/uvicorn src.solution.api:app --host 127.0.0.1 --port 8080
```

Для GPU замените индекс на `cu126`, CLI-устройство на `cuda`, API запускайте с
`DXA_DEVICE=cuda`. Для инференса не нужны scikit-learn, Excel-разметка или тренировочный
манифест. Для разработки установите `requirements-train.txt` вместо runtime-списка.

| Переменная API | По умолчанию | Назначение |
|---|---|---|
| `DXA_MODEL` | `models/radiance`, в образе `/app/models/radiance` | каталог комплекта модели |
| `DXA_DEVICE` | `cpu` | `cpu` или `cuda` |
| `DXA_CPU_THREADS` | `2` | число потоков PyTorch |
| `HF_HUB_OFFLINE` | `1` в образе | исключить сетевые загрузки HF |

CLI имеет `--model`, `--device`, `--output`; принимает файл, папку или ZIP.
Для GPU-пакетного Docker-запуска одного `DXA_DEVICE` недостаточно: передайте CLI
`--device cuda` вместе с `docker run --gpus all`.

## Автономная проверка

```bash
python3 -m src.utils.check_delivery --image radiance:cpu --input data/test
```

Проверка стартует новый контейнер с `--network none`, read-only root и пустым
временным каталогом. Подключается только папка входных снимков, не репозиторий
и не хостовые веса. Через внутренний HTTP проверяются Radiance, ZIP, частичный сбой,
дубли, порядок, ошибки запроса и отсутствие обучающих данных/кода в образе.
Результат — `artifacts/delivery-checks/container.json`.

Полная ML-проверка на сервере с доступной выгрузкой:

```bash
HF_HUB_OFFLINE=1 python -m src.utils.check --device cuda
```

Контейнер нельзя запускать внутри непривилегированного vast.ai-контейнера.
В предоставленной среде API проверяется непосредственно в Python; Docker
собирается и запускается в Docker Desktop. Эти проверки различаются в отчёте.
