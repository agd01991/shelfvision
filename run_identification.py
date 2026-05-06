from __future__ import annotations

import argparse
from pathlib import Path

from src.identification.matcher import run_sku_matching, results_to_dataframe
from src.identification.metrics import evaluate_with_ground_truth, save_identification_metrics
from src.identification.report import save_identification_outputs
from src.identification.track_stabilizer import save_track_summaries, stabilize_results_by_tracks
from src.identification.video_renderer import render_identified_video
from src.identification.visualization import visualize_identification_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShelfVision SKU identification runner")
    parser.add_argument("--predictions", required=True, help="prediction.json или predictions.json после инференса")
    parser.add_argument("--images-dir", default=None, help="Папка исходных изображений, если пути в predictions относительные")
    parser.add_argument("--out-dir", default=r"D:\1Diplom\shelfvision_results\identification", help="Папка для crop-ов, JSON/CSV и визуализаций")
    parser.add_argument("--gallery-csv", default=None, help="CSV с колонками sku_id, sku_name, category, image_path")
    parser.add_argument("--gallery-dir", default=None, help="Папка data/sku_gallery/<sku_id>/*.jpg, если CSV ещё нет")
    parser.add_argument("--gt-csv", default=None, help="Опциональный CSV для оценки: image_name, object_id, true_sku_id")
    parser.add_argument("--threshold", type=float, default=0.65, help="Порог similarity для matched/unknown")
    parser.add_argument("--top-k", type=int, default=3, help="Сколько кандидатов SKU сохранять")
    parser.add_argument("--padding", type=float, default=0.05, help="Padding вокруг bbox при crop extraction")
    parser.add_argument("--use-masks", action="store_true", help="Вырезать crop по mask, если маски есть")
    parser.add_argument("--no-visualize", action="store_true", help="Не сохранять визуализации с подписями SKU")
    parser.add_argument("--visualize-limit", type=int, default=30, help="Сколько изображений визуализировать")
    parser.add_argument("--stabilize-tracks", action="store_true", help="Стабилизировать SKU по track_id из video_predictions.json")
    parser.add_argument("--render-identified-video", action="store_true", help="Собрать identified_output_video.mp4 после идентификации видео")
    parser.add_argument("--video-summary", default=None, help="video_summary.json от run_video_inference.py")
    parser.add_argument("--identified-video-codec", default="mp4v", help="Кодек для identified_output_video.mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.gallery_csv and not args.gallery_dir:
        raise SystemExit("Укажите --gallery-csv или --gallery-dir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_sku_matching(
        predictions_json=args.predictions,
        images_dir=args.images_dir,
        out_dir=out_dir,
        gallery_csv=args.gallery_csv,
        gallery_dir=args.gallery_dir,
        use_masks=args.use_masks,
        threshold=args.threshold,
        top_k=args.top_k,
        padding_ratio=args.padding,
    )

    track_summaries = []
    if args.stabilize_tracks:
        results, track_summaries = stabilize_results_by_tracks(results, video_predictions_json=args.predictions)
        save_track_summaries(track_summaries, out_dir=out_dir)
        # run_sku_matching saves an initial CSV before stabilization; overwrite it with stabilized values.
        results_to_dataframe(results).to_csv(out_dir / "identification_results.csv", index=False)

    metrics = evaluate_with_ground_truth(results, gt_csv=args.gt_csv)
    save_identification_metrics(metrics, out_dir=out_dir)
    save_identification_outputs(
        predictions_json=args.predictions,
        results=results,
        metrics=metrics,
        out_dir=out_dir,
    )

    if not args.no_visualize:
        visualize_identification_results(
            results=results,
            images_dir=args.images_dir,
            out_dir=out_dir,
            limit=args.visualize_limit,
        )

    video_outputs = {}
    if args.render_identified_video:
        video_summary = args.video_summary
        if not video_summary:
            candidate = Path(args.predictions).parent / "video_summary.json"
            if candidate.exists():
                video_summary = str(candidate)
        if not video_summary:
            raise SystemExit("Для сборки identified video укажите --video-summary или положите video_summary.json рядом с predictions")
        video_outputs = render_identified_video(
            video_predictions_json=args.predictions,
            identification_results_json=out_dir / "identification_results.json",
            video_summary_json=video_summary,
            out_dir=out_dir,
            codec=args.identified_video_codec,
        )

    print("=== ShelfVision identification done ===")
    print(f"Results: {out_dir}")
    print(f"Objects: {metrics.get('total_objects', 0)}")
    print(f"Matched: {metrics.get('matched', 0)}")
    print(f"Unknown: {metrics.get('unknown', 0)}")
    if args.stabilize_tracks:
        print(f"Tracks: {len(track_summaries)}")
        print(f"Track summary: {out_dir / 'track_sku_summary.json'}")
    for name, path in video_outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
