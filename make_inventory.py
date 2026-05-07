from pathlib import Path
from datetime import datetime
from collections import defaultdict
import csv
import os


PROJECTS = [
    {
        "name": "project",
        "root": Path("/mnt/c/Users/agd01/Documents/1ДипломМага/Проги/shelfvision"),
    },
    {
        "name": "diskD",
        "root": Path("/mnt/d/1Diplom"),
    },
]


EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".venv_wsl",
    ".venv_wsl_old",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


def should_skip(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def get_extension(file_path: Path) -> str:
    suffix = file_path.suffix
    if not suffix:
        return "no_extension"
    return suffix[1:].lower()


def format_size_kb(size_bytes: int) -> str:
    # Аналог %k KB в find: размер в KB с округлением вверх по блокам
    size_kb = (size_bytes + 1023) // 1024
    return f"{size_kb} KB"


def collect_files(root: Path):
    files = []

    for current_root, dirs, filenames in os.walk(root):
        current_path = Path(current_root)

        # Удаляем исключённые папки из обхода
        dirs[:] = [
            d
            for d in dirs
            if d not in EXCLUDED_DIRS and not d.startswith("_project_inventory")
        ]

        for filename in filenames:
            file_path = current_path / filename
            relative_path = file_path.relative_to(root)

            if should_skip(relative_path):
                continue

        for filename in filenames:
            file_path = current_path / filename
            relative_path = file_path.relative_to(root)

            if should_skip(relative_path):
                continue

            try:
                stat = file_path.stat()
            except OSError:
                continue

            files.append(
                {
                    "relative_path": f"./{relative_path.as_posix()}",
                    "filename": filename,
                    "size_kb": format_size_kb(stat.st_size),
                    "size_bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "extension": get_extension(file_path),
                    "top_folder": relative_path.parts[0]
                    if len(relative_path.parts) > 1
                    else relative_path.name,
                }
            )

    return sorted(files, key=lambda x: x["relative_path"])


def write_files_inventory(files, output_path: Path):
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for item in files:
            writer.writerow(
                [
                    item["relative_path"],
                    item["filename"],
                    item["size_kb"],
                    f"{item['size_bytes']} bytes",
                    item["mtime"],
                    item["extension"] if item["extension"] != "no_extension" else "",
                ]
            )


def write_extensions_summary(files, output_path: Path):
    summary = defaultdict(lambda: {"count": 0, "bytes": 0})

    for item in files:
        ext = item["extension"]
        summary[ext]["count"] += 1
        summary[ext]["bytes"] += item["size_bytes"]

    rows = []
    for ext, data in summary.items():
        rows.append(
            {
                "extension": ext,
                "files_count": data["count"],
                "total_bytes": data["bytes"],
                "total_mb": data["bytes"] / 1024 / 1024,
            }
        )

    rows.sort(key=lambda x: x["extension"])

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["extension", "files_count", "total_bytes", "total_mb"])

        for row in rows:
            writer.writerow(
                [
                    row["extension"],
                    row["files_count"],
                    row["total_bytes"],
                    f"{row['total_mb']:.2f}",
                ]
            )


def write_top_folders_summary(files, output_path: Path):
    summary = defaultdict(lambda: {"count": 0, "bytes": 0})

    for item in files:
        top = item["top_folder"]
        summary[top]["count"] += 1
        summary[top]["bytes"] += item["size_bytes"]

    rows = []
    for folder, data in summary.items():
        rows.append(
            {
                "top_folder": folder,
                "files_count": data["count"],
                "total_bytes": data["bytes"],
                "total_mb": data["bytes"] / 1024 / 1024,
            }
        )

    rows.sort(key=lambda x: x["top_folder"])

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["top_folder", "files_count", "total_bytes", "total_mb"])

        for row in rows:
            writer.writerow(
                [
                    row["top_folder"],
                    row["files_count"],
                    row["total_bytes"],
                    f"{row['total_mb']:.2f}",
                ]
            )


def write_total_size(files, output_path: Path):
    files_count = len(files)
    total_bytes = sum(item["size_bytes"] for item in files)
    total_mb = total_bytes / 1024 / 1024

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"files_count={files_count}\n")
        f.write(f"total_bytes={total_bytes}\n")
        f.write(f"total_mb={total_mb:.2f}\n")


def create_inventory(project_name: str, root: Path):
    if not root.exists():
        print(f"[ERROR] Папка не найдена: {root}")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = root / f"_project_inventory_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Сканирую проект: {root}")
    files = collect_files(root)

    write_files_inventory(
        files,
        output_dir / f"{project_name}_files_inventory_{timestamp}.tsv",
    )

    write_extensions_summary(
        files,
        output_dir / f"{project_name}_extensions_summary_{timestamp}.tsv",
    )

    write_top_folders_summary(
        files,
        output_dir / f"{project_name}_top_folders_summary_{timestamp}.tsv",
    )

    write_total_size(
        files,
        output_dir / f"{project_name}_total_size_{timestamp}.txt",
    )

    total_mb = sum(item["size_bytes"] for item in files) / 1024 / 1024

    print(f"[OK] Инвентаризация готова: {output_dir}")
    print(f"[OK] Файлов: {len(files)}")
    print(f"[OK] Размер: {total_mb:.2f} МБ")
    print()


def main():
    for project in PROJECTS:
        create_inventory(project["name"], project["root"])


if __name__ == "__main__":
    main()
