"""Ориентиры dxa-qc в исходной пиксельной сетке, отдельно от прогноза."""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from src.config import REGION_SPINE, REGION_FEMUR
from src.solution.geometry import spine_axis, hip_field


def foreign_object_mask(img, band):
    # Алгоритм dxa-qc: тонкие яркие структуры вне центральной полосы.
    th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    core = (th > 40) & (img > 150)
    core[:, band[0]:band[1]] = False
    labels, _ = ndi.label((th > 12) & (img > 90))
    sizes = np.bincount(labels.ravel())
    blur = cv2.GaussianBlur(img, (0, 0), 2)
    threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    bones, _ = ndi.label(blur > threshold)
    bone_sizes = np.bincount(bones.ravel())
    keep = []
    for label in set(np.unique(labels[core])) - {0}:
        component = labels == label
        ys, xs = np.where(component)
        length = max(np.ptp(ys), np.ptp(xs)) + 1
        enclosing = max((bone_sizes[b] for b in set(np.unique(bones[component])) - {0}), default=0)
        if length >= 12 and sizes[label] / length <= 8 and enclosing <= 700:
            keep.append(label)
    return np.isin(labels, keep)


def annotate(img, row):
    h, w = img.shape
    overlay = Image.new('RGBA', (w, h))
    draw = ImageDraw.Draw(overlay)
    legend, notes, landmarks = [], [], {}
    region = row.get('anatomical_region')
    found = {s.strip() for s in row.get('violation_type', '').split(';')}
    if min(h, w) < 70 or region not in (REGION_SPINE, REGION_FEMUR):
        return overlay, legend, ['Ориентиры не определены для этого снимка.']

    def color(criterion, label):
        flagged = criterion in found
        value = '#ffb454' if flagged else '#69d4e8'
        legend.append({'label': label, 'flagged': flagged})
        return value

    def line(points, value):
        draw.line(points, fill=value, width=2)
        for x, y in points:
            draw.ellipse((x-2, y-2, x+2, y+2), fill=value)

    if region == REGION_SPINE:
        angles = spine_axis(img, landmarks)
        if not landmarks:
            return overlay, legend, ['Расчётная ось не выделена на этом снимке.']
        m, c = landmarks['axis']
        y0, y1 = landmarks['y_span']
        xmid = m * (y0+y1)/2+c
        for y in range(int(y0), int(y1), 9):
            draw.line([(xmid, y), (xmid, min(y+4, y1))], fill='#ffffff', width=1)
        line([(m*y0+c, y0), (m*y1+c, y1)], color('Не выравнена ось позвоночника',
             f'Расчётная ось: {angles[0]:.1f}° к вертикали кадра'))
        lo, hi = landmarks['band']
        a = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 3)
        bright = a > np.percentile(a[:, lo:hi], 60)
        crests = []
        for left, right in ((0, lo), (hi, w)):
            ys = np.where(bright[h//2:, left:right].sum(1) > 6)[0]
            if len(ys):
                y = int(ys.min()) + h//2
                if .5*h < y < .95*h:
                    crests.append([(left, y), (max(left, right-1), y)])
        if crests:
            crest_color = color('Некорректная укладка', 'Нижние боковые ориентиры укладки')
            for points in crests:
                line(points, crest_color)
        if len(crests) < 2:
            notes.append('Один из нижних боковых ориентиров не найден.' if crests
                         else 'Нижние боковые ориентиры не найдены.')
        if 'Присутствуют посторонние предметы' in found:
            mask = foreign_object_mask(img, (lo, hi))
            if mask.any():
                rgba = np.asarray(overlay).copy()
                rgba[mask] = (255, 180, 84, 170)
                overlay = Image.fromarray(rgba)
                color('Присутствуют посторонние предметы', 'Подозрительные яркие включения')
                notes.append('Подсветка включений — эвристика dxa-qc, возможны совпадения с костными структурами.')
            else:
                notes.append('Посторонние предметы: модель отметила нарушение, расположение не определено.')
    else:
        hip_field(img, landmarks)
        if not landmarks:
            return overlay, legend, ['Контур бедра не выделен на этом снимке.']
        roi_color = color('Некорректная область интереса', 'Отступы поля сканирования')
        top, lat, lesser = landmarks['top'], landmarks['lateral'], landmarks['lesser']
        line([(top[0], 0), top], roi_color)
        line([(0, lat[1]), lat], roi_color)
        if lesser:
            line([lesser, (lesser[0], h-1)], roi_color)
            x, y = lesser
            draw.ellipse((x-6, y-6, x+6, y+6), outline=color('Некорректная укладка', 'Ориентир малого вертела'), width=2)
        else:
            notes.append('Малый вертел не выделен. Оцените ротацию на полном кадре.')
        if landmarks['mirrored']:
            overlay = overlay.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return overlay, legend, notes
