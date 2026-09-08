"""Command-line entry point.

Useful for batch scoring a directory of resumes, for CI smoke tests, and for
inspecting the deterministic scores without starting the server.

    resumeiq analyze resume.pdf --job job.txt
    resumeiq analyze resumes/ --job job.txt --json > results.json
    resumeiq serve --reload
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.extraction.detector import extract
from app.core.llm.service import FeedbackOutcome, generate_feedback
from app.core.scoring.aggregate import AnalysisResult, analyze
from app.errors import ResumeIQError
from app.logging_conf import configure_logging

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="resumeiq", description="ResumeIQ command line")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Score one resume or a directory")
    analyze_parser.add_argument("path", type=Path, help="Resume file or directory of resumes")
    analyze_parser.add_argument("--job", type=Path, help="Job description text file")
    analyze_parser.add_argument("--json", action="store_true", help="Emit JSON instead of a report")
    analyze_parser.add_argument("--ai", action="store_true", help="Include AI coaching")
    analyze_parser.add_argument("-q", "--quiet", action="store_true", help="Errors only")

    serve_parser = subparsers.add_parser("serve", help="Run the API server")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging("ERROR" if getattr(args, "quiet", False) else settings.log_level, "console")

    if args.command == "serve":
        return _serve(args, settings)
    return _analyze(args)


def _serve(args: argparse.Namespace, settings: Any) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install 'resumeiq[dev]'", file=sys.stderr)
        return 1

    uvicorn.run(
        "app.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_config=None,  # keep our structured logging
    )
    return 0


def _analyze(args: argparse.Namespace) -> int:
    targets = _collect(args.path)
    if not targets:
        print(f"No supported resume files found at {args.path}", file=sys.stderr)
        return 1

    job_text = args.job.read_text(encoding="utf-8") if args.job else None
    settings = get_settings()
    results: list[dict[str, Any]] = []
    exit_code = 0

    for target in targets:
        try:
            document = extract(
                target.read_bytes(),
                target.name,
                max_bytes=settings.max_upload_bytes,
                max_chars=settings.max_resume_chars,
            )
            result = analyze(document, job_text=job_text)
        except ResumeIQError as exc:
            print(f"{target.name}: {exc.code}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        feedback = FeedbackOutcome(skipped_reason="not requested")
        if args.ai:
            feedback = asyncio.run(generate_feedback(result))

        if args.json:
            results.append(_as_dict(target, result, feedback))
        else:
            _print_report(target, result, feedback)

    if args.json:
        json.dump(results, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    return exit_code


def _collect(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    return []


def _as_dict(
    target: Path, result: AnalysisResult, feedback: FeedbackOutcome
) -> dict[str, Any]:
    return {
        "file": str(target),
        "overall_score": result.overall_score,
        "band": result.band,
        "dimensions": {d.value: s.score for d, s in result.dimensions.items()},
        "skills": sorted(result.skills),
        "recommendations": [
            {
                "id": r.id,
                "severity": r.severity.value,
                "title": r.title,
                "action": r.action,
                "impact_points": r.impact,
            }
            for r in result.recommendations
        ],
        "match": (
            {
                "required_coverage": result.match.required_coverage,
                "missing_required": [g.name for g in result.match.critical_gaps],
            }
            if result.match
            else None
        ),
        "ai_feedback": (
            feedback.feedback.payload.model_dump() if feedback.feedback else None
        ),
    }


def _print_report(target: Path, result: AnalysisResult, feedback: FeedbackOutcome) -> None:
    width = 74
    print("=" * width)
    print(f"{target.name}  ->  {result.overall_score}/100 ({result.band})")
    print("=" * width)

    for dimension_id, dimension in result.dimensions.items():
        bar = _bar(dimension.score)
        print(f"  {dimension.label:<26} {dimension.score:>5.1f}  {bar}  [{dimension.severity}]")

    if result.match:
        print(f"\n  Required-skill coverage: {result.match.required_coverage:.0%}")
        missing = [g.name for g in result.match.critical_gaps]
        if missing:
            print(f"  Missing required skills: {', '.join(missing[:10])}")

    print("\n  Top recommendations")
    print("  " + "-" * (width - 4))
    for recommendation in result.recommendations[:6]:
        print(f"  [{recommendation.severity.upper():<8}] +{recommendation.impact:>5.2f} pts  "
              f"{recommendation.title}")
        print(f"             {recommendation.action}")

    if feedback.feedback:
        payload = feedback.feedback.payload
        print(f"\n  AI verdict: {payload.headline}")
        for fix in payload.priority_fixes[:3]:
            print(f"   - {fix.target}: {fix.fix}")
    elif feedback.skipped_reason and feedback.skipped_reason != "not requested":
        print(f"\n  (AI coaching unavailable: {feedback.skipped_reason})")
    print()


def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score / 100 * width))
    return "#" * filled + "." * (width - filled)


if __name__ == "__main__":
    raise SystemExit(main())
