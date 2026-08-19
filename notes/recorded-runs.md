# Reproducible experiments

Use the following categories in reports:

| Category | What it establishes |
| --- | --- |
| Unit test | A specific software invariant on specified inputs |
| Reference/oracle evaluation | Baseline or upper-bound behavior in a synthetic environment |
| Synthetic response replay | Evaluation/routing behavior on fixture responses; not provider performance |
| Learned local experiment | Performance of a trained model on the stated held-out split |
| Real-model experiment | Performance of the identified checkpoint/provider on the stated corpus |

## Record before comparing

Record the code revision, Python and dependency versions, dataset hash, split rule, random seed, model identity and revision, prompt/configuration version, inference settings, and hardware. Record actual price configuration if reporting monetary cost. Unknown or simulated price is not a measured invoice.

State the numerator and denominator for each rate. Preserve per-case failures so aggregate improvements cannot hide a broken slice. Use independent ground truth: a simulator, parser, explicit event history, relevance labels, or reviewed answers.

## Avoid leakage

Development examples tune the implementation. Validation data selects settings. Held-out evaluation measures the final choice. Split by composition, template, task family, or time where random rows would share the same underlying answer pattern. Do not tune repeatedly against the held-out set and continue calling it unseen.

## Compare fairly

Use identical evaluation cases for competing methods. State each method's context and inference budget. Report correctness together with latency and cost where applicable. Separate syntactic validity from semantic success, retrieval quality from answer accuracy, and recovery behavior from model capability.

## Publish failures

Include incorrect outputs, abstentions, timeouts, invalid tool calls, and incomplete runs in the denominator according to an explicit rule. Do not drop failed runs to improve scores. A run that could not start is `not_run` or `error`, never a perfect score or a silently substituted fixture result.

## Optional model work

Training scripts are implementations, not proof that training has completed. Preserve the checkpoint's model card, preprocessing and tokenizer settings, dataset split, and evaluation command after a real run. Keep large weights outside the Git history; link the appropriate artifact once available.
