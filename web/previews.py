"""Временные PNG из того же полного кадра, который читает модель."""

import base64
from io import BytesIO
import zipfile

import pydicom
from PIL import Image

from src.solution.dicom import pixels
from .annotations import annotate

MAX_FILE = 32 * 1024**2
MAX_EXPANDED = 512 * 1024**2
MAX_PREVIEWS = 32 * 1024**2


def build_previews(source, rows):
    unavailable = {"message": "Снимок недоступен для просмотра. Результат проверки сохранён."}
    result = [unavailable.copy() for _ in rows]
    source.seek(0)
    try:
        with zipfile.ZipFile(source, metadata_encoding="cp866") as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            # Порядок и полный путь сохраняют связь даже при одинаковых именах.
            if (len(members) != len(rows) or len(members) > 10000
                    or sum(m.file_size for m in members) > MAX_EXPANDED
                    or any(m.filename != r["path_to_study"] for m, r in zip(members, rows))):
                return result
            used = 0
            for index, (member, row) in enumerate(zip(members, rows)):
                if row["processing_status"] != "Success":
                    result[index] = {"message": "Файл не обработан. Загрузите исправный DICOM для просмотра."}
                    continue
                if used >= MAX_PREVIEWS:
                    result[index] = {"message": "Лимит превью пакета достигнут. Загрузите этот DICOM отдельно."}
                    continue
                if member.file_size > MAX_FILE or member.flag_bits & 1:
                    continue
                try:
                    with archive.open(member) as stream:
                        ds = pydicom.dcmread(stream)
                        frame = pixels(ds)
                        image = Image.fromarray(frame)
                    original = image.size
                    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    output = BytesIO()
                    image.save(output, format="PNG")
                    data = output.getvalue()
                    if used + len(data) > MAX_PREVIEWS:
                        result[index] = {"message": "Лимит превью пакета достигнут. Загрузите этот DICOM отдельно."}
                        continue
                    try:
                        overlay, legend, notes = annotate(frame, row)
                        annotation = {"legend": legend, "annotation_notes": notes}
                        if overlay.getbbox():
                            overlay = overlay.resize(image.size, Image.Resampling.LANCZOS)
                            overlay_output = BytesIO()
                            overlay.save(overlay_output, format="PNG")
                            overlay_data = overlay_output.getvalue()
                            if used + len(data) + len(overlay_data) <= MAX_PREVIEWS:
                                annotation["overlay"] = "data:image/png;base64," + base64.b64encode(overlay_data).decode("ascii")
                                used += len(overlay_data)
                            else:
                                annotation = {"annotation_notes": ["Лимит разметки пакета достигнут. Откройте DICOM отдельно."]}
                    except Exception:
                        annotation = {"annotation_notes": ["Разметка недоступна. Снимок и результат проверки сохранены."]}
                    used += len(data)
                    result[index] = {"image": "data:image/png;base64," + base64.b64encode(data).decode("ascii"),
                                     "width": original[0], "height": original[1],
                                     "reduced": image.size != original, **annotation}
                except Exception:
                    # Ошибка просмотра не отменяет уже полученный CSV.
                    continue
    except (zipfile.BadZipFile, ValueError, OSError):
        pass
    return result
