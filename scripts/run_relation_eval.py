"""Phase 4: score support / attack relation classification against your own annotations.

Run from the project root, after filling data/annotation/pairs.csv:
    python scripts/run_relation_eval.py --target "the minimum wage"

Rows you left blank are skipped. Prints precision, recall and F1 for Support / Attack / None, a confusion
matrix, Cohen's kappa between the model and you, and accuracy split by same-speaker vs different-speaker pairs
(rebuttals are usually between different speakers, so that split is useful for error analysis).

This scores the relation step on its own: the model sees your sentences exactly as written in the sheet.
Every prediction is saved to results/annotated_relations.jsonl.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.annotation import RELATION_LABELS, interpret_kappa, load_pair_sheet  # noqa: E402
from argus.evaluation import Progress, predict_relations, write_json, write_jsonl  # noqa: E402
from argus.llm import make_ready_client  # noqa: E402
from argus.metrics import classification_report, cohen_kappa, format_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=ROOT / "data" / "annotation" / "pairs.csv")
    ap.add_argument("--target", required=True, help='the policy under debate, e.g. "the minimum wage"')
    ap.add_argument("--model", default=None, help="override the model in config.yaml")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    try:
        pairs, blank = load_pair_sheet(args.pairs)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))                      # a plain message, not a traceback
    if not pairs:
        sys.exit(f"No annotated rows in {args.pairs} ({blank} blank). Fill 'gold_relation' first.")
    llm = make_ready_client(args.model)
    print(f"Model: {llm.model} | annotated pairs: {len(pairs)} (skipped {blank} blank)")

    progress = Progress(len(pairs), "relations")
    rows = predict_relations(llm, args.target, pairs, progress)
    progress.close()

    gold, pred = [r["gold"] for r in rows], [r["pred"] for r in rows]
    report = classification_report(gold, pred, RELATION_LABELS)
    kappa = cohen_kappa(gold, pred)
    print("\n" + format_report(report))
    print(f"\nCohen's kappa, model vs your annotation: {kappa:.3f} ({interpret_kappa(kappa)})")

    for name, same in (("same speaker", True), ("different speakers", False)):
        subset = [r for r in rows if bool(r["speaker_a"]) and (r["speaker_a"] == r["speaker_b"]) == same]
        if subset:
            acc = sum(r["gold"] == r["pred"] for r in subset) / len(subset)
            print(f"Accuracy, {name}: {acc:.3f} (n={len(subset)})")

    wrong = [r for r in rows if r["gold"] != r["pred"]]
    if wrong:
        print(f"\nDisagreements ({len(wrong)}):")
        for r in wrong[:12]:
            print(f"  [{r['pair_id']}] you: {r['gold']:<7} model: {r['pred']:<7} | "
                  f"A: {r['sentence_a'][:42]} | B: {r['sentence_b'][:42]}")
        if len(wrong) > 12:
            print(f"  ... and {len(wrong) - 12} more in the results file")

    write_jsonl(rows, args.out_dir / "annotated_relations.jsonl")
    write_json({"model": llm.model, "report": report, "kappa_model_vs_human": kappa},
               args.out_dir / "annotated_relations.json")
    print(f"\nSaved {args.out_dir / 'annotated_relations.jsonl'} and .json")


if __name__ == "__main__":
    main()
