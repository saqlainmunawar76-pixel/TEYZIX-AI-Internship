"""
main.py – Command-line interface for the AI Document Summarization System
Task ID : AI-INT-1  |  Domain: AI / NLP
TEYZIX CORE Internship – June Batch 2026
"""

import os
import sys
import argparse

from summarizer import load_text, extractive_summarize, export_summary


# ══════════════════════════════════════════════════════════════════════════════
# Helper: pretty-print result
# ══════════════════════════════════════════════════════════════════════════════

def print_result(result: dict) -> None:
    sep  = "=" * 60
    dash = "-" * 60

    print(f"\n{sep}")
    print("  AI DOCUMENT SUMMARIZATION SYSTEM — TEYZIX CORE")
    print(sep)
    print(f"  Method      : {result['method'].upper()}")
    print(f"  Orig. words : {result['original_word_count']}")
    print(f"  Summ. words : {result['summary_word_count']}")
    print(f"  Compression : {result['compression_ratio']}%")
    print(f"  Sentences   : {result['summary_sentence_count']} of {result['original_sentence_count']} used")
    print(f"\n{dash}")
    print("  SUMMARY")
    print(dash)
    print(f"\n{result['summary']}\n")
    print(dash)
    print("  TOP KEYWORDS")
    print(dash)
    for word, freq in result["top_keywords"]:
        bar = "█" * min(freq, 30)
        print(f"  {word:<18} {bar} ({freq})")
    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Interactive mode
# ══════════════════════════════════════════════════════════════════════════════

def interactive_mode() -> None:
    print("\n" + "=" * 60)
    print("  AI-POWERED DOCUMENT SUMMARIZATION SYSTEM")
    print("  TEYZIX CORE Internship | Task AI-INT-1")
    print("=" * 60)

    print("\nSelect input source:")
    print("  1. Enter text directly")
    print("  2. Load from .txt file")
    print("  3. Load from .pdf file")

    while True:
        choice = input("\nYour choice (1/2/3): ").strip()
        if choice in ("1", "2", "3"):
            break
        print("[!] Invalid choice. Enter 1, 2, or 3.")

    # ── Load text ──────────────────────────────────────────────
    try:
        if choice == "1":
            print("\nPaste your text below. Press Enter twice when done:")
            lines = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            text = "\n".join(lines).strip()
            if not text:
                print("[!] No text provided. Exiting.")
                return

        elif choice == "2":
            path = input("Enter .txt file path: ").strip().strip('"')
            text = load_text(path, source_type="txt")

        else:
            path = input("Enter .pdf file path: ").strip().strip('"')
            text = load_text(path, source_type="pdf")

    except (FileNotFoundError, ValueError, ImportError) as err:
        print(f"[Error] {err}")
        return

    # ── Config ────────────────────────────────────────────────
    print("\nSummarization method:")
    print("  1. TF-IDF  (default, best accuracy)")
    print("  2. Frequency-based  (fast)")
    print("  3. Combined  (TF-IDF + Frequency)")

    method_map = {"1": "tfidf", "2": "frequency", "3": "combined"}
    m_choice = input("Method (1/2/3, default=1): ").strip() or "1"
    method = method_map.get(m_choice, "tfidf")

    try:
        n = int(input("Number of sentences in summary (default=3): ").strip() or "3")
    except ValueError:
        n = 3

    # ── Run ───────────────────────────────────────────────────
    print("\n[*] Processing…")
    try:
        result = extractive_summarize(text, num_sentences=n, method=method)
    except ValueError as err:
        print(f"[Error] {err}")
        return

    print_result(result)

    # ── Export ────────────────────────────────────────────────
    save = input("Export summary to file? (y/n, default=y): ").strip().lower() or "y"
    if save == "y":
        out_dir = "outputs"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "summary_output.txt")
        export_summary(result, out_path, fmt="txt")


# ══════════════════════════════════════════════════════════════════════════════
# CLI argument mode
# ══════════════════════════════════════════════════════════════════════════════

def cli_mode(args) -> None:
    # ── Load ──────────────────────────────────────────────────
    try:
        if args.text:
            text = args.text
        elif args.file:
            text = load_text(args.file, source_type="auto")
        else:
            print("[!] Provide --text or --file.")
            sys.exit(1)
    except (FileNotFoundError, ValueError, ImportError) as err:
        print(f"[Error] {err}")
        sys.exit(1)

    # ── Summarize ─────────────────────────────────────────────
    try:
        result = extractive_summarize(text,
                                      num_sentences=args.sentences,
                                      method=args.method)
    except ValueError as err:
        print(f"[Error] {err}")
        sys.exit(1)

    print_result(result)

    # ── Export ────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        export_summary(result, args.output, fmt="txt")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Document Summarization System – TEYZIX CORE"
    )
    parser.add_argument("--file",      type=str, help="Path to .txt or .pdf input file")
    parser.add_argument("--text",      type=str, help="Raw text string to summarize")
    parser.add_argument("--sentences", type=int, default=3,
                        help="Number of sentences in summary (default: 3)")
    parser.add_argument("--method",    type=str, default="tfidf",
                        choices=["tfidf", "frequency", "combined"],
                        help="Scoring method (default: tfidf)")
    parser.add_argument("--output",    type=str, help="Output file path (.txt)")

    args = parser.parse_args()

    # If no args given → interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        cli_mode(args)


if __name__ == "__main__":
    main()
