from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "youtube_short_snapshots"


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", normalized.lower()).strip("_")
    return cleaned or "source"


def _looks_like_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _youtube_label_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    parts = [part for part in parsed.path.split("/") if part]

    if host.endswith("youtu.be") and parts:
        candidate = parts[0]
    elif "youtube.com" in host:
        if len(parts) >= 2 and parts[0] == "shorts":
            candidate = parts[1]
        elif len(parts) >= 2 and parts[0] == "embed":
            candidate = parts[1]
        else:
            candidate = parse_qs(parsed.query).get("v", [parts[-1] if parts else host])[0]
    else:
        candidate = parts[-1] if parts else host or "source"
    return _slugify(candidate)


def _label_from_source(source: str) -> str:
    if _looks_like_url(source):
        return _youtube_label_from_url(source)

    path = Path(source).expanduser()
    return _slugify(path.stem or path.name)


def _yt_dlp_prefix() -> list[str]:
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]

    executable = shutil.which("yt-dlp")
    if executable:
        return [executable]

    raise RuntimeError(
        "yt-dlp est requis pour telecharger une URL YouTube. "
        "Installe-le avec `pip install -r requirements.txt` ou passe un fichier video local."
    )


def _run_command(command: list[str], *, label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "(aucune sortie d'erreur)"
        raise RuntimeError(f"{label} a echoue.\nCommande: {' '.join(command)}\nErreur: {stderr}")
    return result


def _parse_frame_rate(raw_value: str) -> float:
    value = raw_value.strip()
    if not value or value.upper() == "N/A":
        return 0.0

    if "/" in value:
        numerator_text, denominator_text = value.split("/", 1)
        numerator = float(numerator_text)
        denominator = float(denominator_text)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    return float(value)


def _probe_frame_rate(video_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 30.0

    for stream_rate_key in ("avg_frame_rate", "r_frame_rate"):
        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            f"stream={stream_rate_key}",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        result = _run_command(command, label="ffprobe")
        try:
            frame_rate = _parse_frame_rate(result.stdout)
        except ValueError:
            frame_rate = 0.0
        if frame_rate > 0:
            return frame_rate

    return 30.0


def _download_youtube_video(url: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    command = [
        *_yt_dlp_prefix(),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-P",
        str(download_dir),
        "-o",
        "source.%(ext)s",
        "--print",
        "after_move:filepath",
        url,
    ]
    result = _run_command(command, label="yt-dlp")
    printed_paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    for candidate_text in reversed(printed_paths):
        candidate = Path(candidate_text)
        if candidate.exists():
            return candidate

        if not candidate.is_absolute():
            local_candidate = download_dir / candidate
            if local_candidate.exists():
                return local_candidate

    fallback = sorted(download_dir.glob("source.*"))
    if len(fallback) == 1 and fallback[0].is_file():
        return fallback[0]

    raise RuntimeError("yt-dlp a termine, mais aucun fichier video telecharge n'a ete trouve.")


def _ensure_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError(f"Le chemin de sortie existe deja mais ce n'est pas un dossier: {output_dir}")
        if any(output_dir.iterdir()):
            raise RuntimeError(
                f"Le dossier de sortie existe deja et n'est pas vide: {output_dir}\n"
                "Choisis un autre dossier de sortie ou supprime son contenu avant de relancer."
            )
        return

    output_dir.mkdir(parents=True, exist_ok=True)


def _extract_images(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float,
    *,
    all_frames: bool,
    tail_seconds: float | None,
) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg est requis pour extraire les snapshots, mais il n'est pas disponible.")

    _ensure_empty_output_dir(output_dir)

    output_pattern = output_dir / "frame_%05d.png"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if tail_seconds is not None:
        command.extend(["-sseof", f"-{tail_seconds:.10g}"])
    command.extend([
        "-i",
        str(video_path),
    ])
    if all_frames:
        command.extend(["-vsync", "0", "-start_number", "1"])
    else:
        fps_value = 1.0 / interval_seconds
        command.extend(["-vf", f"fps={fps_value:.10g}"])
    command.append(str(output_pattern))
    _run_command(command, label="ffmpeg")

    frames = sorted(output_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("Aucune image n'a ete generee.")
    return frames


def _build_manifest(
    *,
    source_kind: str,
    source_value: str,
    source_label: str,
    source_path: Path | None,
    output_dir: Path,
    extraction_mode: str,
    tail_seconds: float | None,
    interval_seconds: float,
    frame_files: list[Path],
) -> dict[str, object]:
    source: dict[str, object] = {
        "kind": source_kind,
        "value": source_value,
        "label": source_label,
    }
    if source_path is not None:
        source["path"] = str(source_path)

    if source_kind == "youtube_url":
        source["downloaded_with"] = "yt-dlp"

    return {
        "tool": "video_tools/extract_youtube_short_snapshots.py",
        "created_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir),
        "extraction_mode": extraction_mode,
        "tail_seconds": tail_seconds,
        "interval_seconds": interval_seconds,
        "frame_count": len(frame_files),
        "image_format": "png",
        "source": source,
        "frames": [
            {
                "frame_number": index + 1,
                "timestamp_seconds": round(index * interval_seconds, 6),
                "file_name": frame.name,
            }
            for index, frame in enumerate(frame_files)
        ],
    }


def _default_output_dir(source_label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return DEFAULT_OUTPUT_ROOT / f"{source_label}_{timestamp}"


def _resolve_local_source(source: str) -> Path:
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Fichier video introuvable: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Le chemin pointe vers un dossier, pas vers une video: {path}")
    return path.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extrait des snapshots reguliers ou toutes les frames d'une video YouTube ou locale."
    )
    parser.add_argument(
        "source",
        help="URL YouTube (shorts, watch, youtu.be) ou chemin vers un fichier video local.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Intervalle entre snapshots, en secondes. Defaut: 1.0.",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Extrait chaque frame decodee de la video au lieu d'un snapshot a intervalle fixe.",
    )
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=None,
        help="N'extrait que les N dernieres secondes de la video avant de la decouper. Defaut: tout le clip.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Dossier de sortie pour les PNG et le manifest. "
            "Par defaut, un dossier horodate dans data/processed/youtube_short_snapshots."
        ),
    )
    args = parser.parse_args(argv)

    if args.interval <= 0:
        parser.error("--interval doit etre strictement superieur a 0.")
    if args.tail_seconds is not None and args.tail_seconds <= 0:
        parser.error("--tail-seconds doit etre strictement superieur a 0.")

    source_label = _label_from_source(args.source)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(source_label).resolve()
    )

    if _looks_like_url(args.source):
        source_kind = "youtube_url"
        with tempfile.TemporaryDirectory(prefix=f"yt_short_{source_label}_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            downloaded_path = _download_youtube_video(args.source, temp_dir)
            effective_interval_seconds = (
                1.0 / _probe_frame_rate(downloaded_path) if args.all_frames else args.interval
            )
            frame_files = _extract_images(
                downloaded_path,
                output_dir,
                effective_interval_seconds,
                all_frames=args.all_frames,
                tail_seconds=args.tail_seconds,
            )
            manifest = _build_manifest(
                source_kind=source_kind,
                source_value=args.source,
                source_label=source_label,
                source_path=None,
                output_dir=output_dir,
                extraction_mode="all_frames" if args.all_frames else "interval_snapshots",
                tail_seconds=args.tail_seconds,
                interval_seconds=effective_interval_seconds,
                frame_files=frame_files,
            )
    else:
        source_kind = "local_file"
        local_path = _resolve_local_source(args.source)
        effective_interval_seconds = (
            1.0 / _probe_frame_rate(local_path) if args.all_frames else args.interval
        )
        frame_files = _extract_images(
            local_path,
            output_dir,
            effective_interval_seconds,
            all_frames=args.all_frames,
            tail_seconds=args.tail_seconds,
        )
        manifest = _build_manifest(
            source_kind=source_kind,
            source_value=args.source,
            source_label=source_label,
            source_path=local_path,
            output_dir=output_dir,
            extraction_mode="all_frames" if args.all_frames else "interval_snapshots",
            tail_seconds=args.tail_seconds,
            interval_seconds=effective_interval_seconds,
            frame_files=frame_files,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.all_frames:
        print(f"[video_tools] {len(frame_files)} frames extraites -> {output_dir}")
    else:
        print(f"[video_tools] {len(frame_files)} snapshots extraits -> {output_dir}")
    print(f"[video_tools] manifest -> {manifest_path}")
    if args.all_frames:
        print(f"[video_tools] mode -> all_frames")
        print(f"[video_tools] intervalle estime -> {effective_interval_seconds:.6f} s")
    else:
        print(f"[video_tools] intervalle -> {args.interval} s")
    if args.tail_seconds is not None:
        print(f"[video_tools] queue video -> {args.tail_seconds:.3f} s")
    if frame_files:
        print(f"[video_tools] premier snapshot -> {frame_files[0].name}")
        print(f"[video_tools] dernier snapshot -> {frame_files[-1].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
