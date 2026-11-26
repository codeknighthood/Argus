# Annotation guide: Claim, Premise, None

Use this when labelling `gold_component` in the annotation sheet. Two annotators following the same rules is what
makes Cohen's kappa meaningful. These definitions are the same ones the extraction prompt
(`argus/prompts/extract_v1.txt`) gives the model, so the model is judged by the rules it was given.

Label **one sentence at a time**. Read the `context` column only to understand what the sentence refers to.

| Label | Use it when the sentence is... | Example (target: the minimum wage) |
|---|---|---|
| **Claim** | a standpoint on the target, or on another speaker's position, that someone could dispute | "I support raising it." / "A gradual increase would be far safer." |
| **Premise** | a reason, evidence, example or consequence offered to back up or argue against a claim | "Small businesses run on thin margins." |
| **None** | not an argument: greetings, procedure, questions, background, filler | "Thank you all." / "How many studies were rural?" |

## Rules for the awkward cases

1. **A plain fact is a Premise only if it is being used as a reason.** "Unemployment is 4 percent" with no
   bearing on the debate is None; the same fact offered against a wage rise is a Premise.
2. **Questions are None**, even pointed ones. (Record that you were tempted; it is useful for error analysis.)
3. **Agreement or disagreement with someone ("I disagree with that.")** is a Claim when it takes a position,
   even if it is short.
4. **A sentence that contains both a standpoint and a reason** gets the label of its main point; if you
   truly cannot choose, pick Claim and note it in a `notes` column (add one if you like; it is ignored).
5. **Do not label by sentiment.** A negative-sounding sentence can be a Premise for the policy.
6. **When unsure between Premise and None, ask:** is it offered as a reason? If not, None.

## Relations: Support, Attack, None (`pairs.csv`, column `gold_relation`)

Each row is a pair of sentences. **A is the earlier sentence, B the later one.** Answer one question:
**how does B relate to A?**

| Label | Use it when B... | Example (A is "Full-time workers should not live in poverty.") |
|---|---|---|
| **Support** | backs A up: gives a reason or evidence for A, or agrees with A's position | B: "A living wage lifts families out of poverty." |
| **Attack** | argues against A: rebuts it, says why it is wrong, or takes the opposing position | B: "I disagree. Small firms would have to cut jobs." |
| **None** | is not directly connected to A. Sharing the topic is not enough | B: "Let us hear from the audience." |

Rules for the awkward cases:

1. **Judge B toward A, not toward the policy.** Two arguments that are both against the policy can still Support
   each other, and two that are both for it can still Attack each other (one rebuts the other's reasoning).
2. **B must respond to or build on A.** If B would read the same without A, it is None.
3. **Questions and procedure** ("Let us hear from the audience.") are None.
4. **Disagreeing in a different way** ("I disagree, but for another reason") is still an Attack on A.
5. **Pairs from the same speaker** are usually Support (adding a reason) or None. Mark Attack only if the speaker
   really contradicts themselves or corrects an earlier point.
6. **Do not use the other sentences' labels.** Each pair is judged on its own two sentences; read the
   neighbouring rows only to understand what a sentence refers to.

## Procedure for two annotators

1. Generate the sheets once:
   `python scripts/make_annotation_sheet.py --transcript <file>` (writes `components.csv` and `pairs.csv`).
2. Copy them. Each annotator fills `gold_component` and `gold_relation` independently in Excel, without discussing.
3. Compare:
   `python scripts/annotator_agreement.py A_components.csv B_components.csv`
   `python scripts/annotator_agreement.py A_pairs.csv B_pairs.csv --column gold_relation`
4. Discuss the listed disagreements, settle one final copy of each sheet, and use those for
   `run_annotated_eval.py` and `run_relation_eval.py`.
5. Report the kappa from step 3 (before discussion) in the evaluation chapter.

Tip: the pairs sheet has about three rows per sentence. If that is too much work, regenerate it with
`--kind pairs --pair-window 2 --force` before anyone starts annotating.

Rough reading of kappa: below 0.40 is weak, 0.40 to 0.60 moderate, 0.60 to 0.80 substantial, above 0.80 almost perfect.
