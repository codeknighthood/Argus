"""Phase 3: score claim / premise extraction against your own annotations.

Run from the project root, after filling data/annotation/components.csv:
    python scripts/run_annotated_eval.py --target "the minimum wage"

Rows you left blank are skipped, so you can score a partly annotated sheet.
Prints precision, recall, F1 and a confusion matrix for Claim / Premise / None, plus Cohen's kappa between
the model and you. Every prediction is saved to results/annotated_components.jsonl for error analysis.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.annotation import COMPONENT_LABELS, interpret_kappa, load_component_sheet  # noqa: E402
from argus.evaluation import Progress, write_json, write_jsonl  # noqa: E402
from argus.extract import extract_argument  # noqa: E402
from argus.llm import make_ready_client  # noqa: E402
from argus.metrics import classification_report, cohen_kappa, format_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", type=Path, default=ROOT / "data" / "annotation" / "components.csv")
    ap.add_argument("--target", required=True, help='the policy under debate, e.g. "the minimum wage"')
    ap.add_argument("--model", default=None, help="override the model in config.yaml")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    try:
        rows, blank = load_component_sheet(args.components)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))                      # a plain message, not a traceback
    if not rows:
        sys.exit(f"No annotated rows in {args.components} ({blank} blank). Fill 'gold_component' first.")
    llm = make_ready_client(args.model)
    print(f"Model: {llm.model} | annotated rows: {len(rows)} (skipped {blank} blank)")

    progress = Progress(len(rows), "extract")
    out = []
    for row in rows:
        entry = {"index": row.get("index"), "speaker": row.get("speaker", ""), "sentence": row["sentence"],
                 "gold": row["gold"], "pred": None, "identified_argument": "", "explanation": "", "error": None}
        try:
            res = extract_argument(llm, args.target, row["sentence"], row.get("context", ""))
            entry.update(pred=res.component.value, identified_argument=res.identified_argument,
                         explanation=res.explanation)
        except Exception as e:
            entry.update(pred="ERROR", error=str(e)[:300])
        out.append(entry)
        progress.update()
    progress.close()

    gold, pred = [r["gold"] for r in out], [r["pred"] for r in out]
    report = classification_report(gold, pred, COMPONENT_LABELS)
    kappa = cohen_kappa(gold, pred)
    print("\n" + format_report(report))
    print(f"\nCohen's kappa, model vs your annotation: {kappa:.3f} ({interpret_kappa(kappa)})")

    wrong = [r for r in out if r["gold"] != r["pred"]]
    if wrong:
        print(f"\nDisagreements ({len(wrong)}):")
        for r in wrong[:15]:
            print(f"  [{r['index']}] you: {r['gold']:<8} model: {r['pred']:<8} | {r['sentence'][:80]}")
        if len(wrong) > 15:
            print(f"  ... and {len(wrong) - 15} more in the results file")

    write_jsonl(out, args.out_dir / "annotated_components.jsonl")
    write_json({"model": llm.model, "report": report, "kappa_model_vs_human": kappa},
               args.out_dir / "annotated_components.json")
    print(f"\nSaved {args.out_dir / 'annotated_components.jsonl'} and .json")


if __name__ == "__main__":
    main()
