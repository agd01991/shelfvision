import argparse
from pathlib import Path
import cv2
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issues", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    issues = json.loads(Path(args.issues).read_text(encoding="utf-8"))
    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
    img_map = {im["id"]: im for im in coco["images"]}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clipped = issues["issues"]["clipped"][: args.n]
    for i, item in enumerate(clipped):
        image_id = item["image_id"]
        im = img_map.get(image_id)
        if not im:
            continue

        fn = Path(im["file_name"]).name
        p = Path(args.image_root) / fn
        img = cv2.imread(str(p))
        if img is None:
            continue

        before = item["before"]
        after = item["after"]

        def draw(b, color):
            x, y, w, h = b
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv2.rectangle(img, p1, p2, color, 2)

        draw(before, (0, 0, 255))  # red
        draw(after, (0, 255, 0))  # green

        cv2.putText(
            img,
            f"id={image_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.imwrite(str(out_dir / f"{i:02d}_img{image_id}.jpg"), img)


if __name__ == "__main__":
    main()
