from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


Number = Union[int, float]


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


def is_number_list(x: Any) -> bool:
    return isinstance(x, list) and len(x) > 0 and all(is_number(v) for v in x)


def is_point_pair_list(x: Any) -> bool:
    # [[x,y], [x,y], ...]
    return (
        isinstance(x, list)
        and len(x) > 0
        and all(isinstance(p, (list, tuple)) and len(p) == 2 and is_number(p[0]) and is_number(p[1]) for p in x)
    )


def flatten_pairs(pairs: List[List[Number]]) -> List[float]:
    flat: List[float] = []
    for x, y in pairs:
        flat.append(float(x))
        flat.append(float(y))
    return flat


def fix_one_segmentation(seg: Any) -> Tuple[Any, bool]:
    """
    Возвращает (исправленный_seg, changed)
    Поддерживаем случаи:
      - seg = [x1,y1,x2,y2,...]                  -> [[...]]
      - seg = [[x1,y1,x2,y2,...], [...]]         -> оставляем как есть
      - seg = [[x,y],[x,y],...]                  -> [[x1,y1,x2,y2,...]]
      - seg = [[[x,y],...]]                      -> [[x1,y1,x2,y2,...]]
      - seg = dict (RLE)                         -> оставляем
    """
    if seg is None:
        return None, False

    # RLE (ок для detectron2, если mask_format="bitmask", но мы всё равно оставим)
    if isinstance(seg, dict):
        return seg, False

    if not isinstance(seg, list) or len(seg) == 0:
        return seg, False

    changed = False

    # Case: flat list of numbers -> wrap
    if is_number_list(seg):
        return [seg], True

    # Case: list of point pairs -> flatten and wrap
    if is_point_pair_list(seg):
        return [flatten_pairs(seg)], True

    # Case: seg = [ polygon1, polygon2, ... ]
    # polygon may be list-of-numbers OR list-of-pointpairs
    if all(is_number_list(p) for p in seg):
        # already list-of-polygons
        return seg, False

    if all(is_point_pair_list(p) for p in seg):
        return [flatten_pairs(p) for p in seg], True

    # Case: single nested
    if len(seg) == 1 and isinstance(seg[0], list):
        inner = seg[0]
        if is_number_list(inner):
            return [inner], True
        if is_point_pair_list(inner):
            return [flatten_pairs(inner)], True
        if all(is_number_list(p) for p in inner):
            return inner, True
        if all(is_point_pair_list(p) for p in inner):
            return [flatten_pairs(p) for p in inner], True

    # unknown structure -> leave (will likely fail later, so mark changed False)
    return seg, False


def valid_polygon(poly: List[float]) -> bool:
    # COCO polygon: even length, >= 6
    return isinstance(poly, list) and len(poly) >= 6 and (len(poly) % 2 == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--drop_bad", action="store_true", help="drop annotations with invalid segmentation after fix")
    args = ap.parse_args()

    inp = Path(args.in_json).resolve()
    outp = Path(args.out_json).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)

    coco = json.loads(inp.read_text(encoding="utf-8"))
    anns = coco.get("annotations", [])

    changed_cnt = 0
    dropped_cnt = 0
    bad_cnt = 0

    new_anns = []
    for a in anns:
        seg = a.get("segmentation", None)
        seg2, changed = fix_one_segmentation(seg)
        if changed:
            changed_cnt += 1

        # validate if polygon
        ok = True
        if isinstance(seg2, list):
            # list-of-polygons expected
            # allow empty list? (not for instance seg)
            if len(seg2) == 0:
                ok = False
            else:
                # each element should be polygon list-of-numbers
                for poly in seg2:
                    if not is_number_list(poly) or not valid_polygon(poly):
                        ok = False
                        break

        a2 = dict(a)
        a2["segmentation"] = seg2

        if not ok:
            bad_cnt += 1
            if args.drop_bad:
                dropped_cnt += 1
                continue

        new_anns.append(a2)

    coco["annotations"] = new_anns
    outp.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    print(f"[fix_coco_segmentation] in={inp}")
    print(f"[fix_coco_segmentation] out={outp}")
    print(f"[fix_coco_segmentation] anns_total={len(anns)} changed={changed_cnt} bad={bad_cnt} dropped={dropped_cnt}")


if __name__ == "__main__":
    main()