# MiniReasoner Arithmetic Post Training Plan Book

**Decision · 24 November 2026.** SimpleStories pretraining finished at **24,000 total optimizer steps**. Keep the 32M model and its 1,024-token tokenizer. Then teach the base model to answer arithmetic questions, measure what supervised training achieves, and use reinforcement learning only where a correct answer can be verified and there is room to improve.

## 1 Finish pretraining and choose the base

On 24 November the VM finished the scheduled 24,000 steps and confirmed the final checkpoint upload to Azure. The training column below is the mean over the preceding 500 steps. Each validation check scores a random half of the held-out story tokens. [Databricks run history](https://dbc-327f4e74-291b.cloud.databricks.com/ml/experiments/3756008463948203/runs/655ddf770790468d988bbb0375fd5aa4)

| Step | Mean training loss | Validation loss |
| ---: | ---: | ---: |
| 12,000 | 1.4390 | 1.4377 |
| 13,000 | 1.4272 | 1.4243 |
| 14,000 | 1.4186 | 1.4152 |
| 16,000 | 1.3984 | 1.3973 |
| 20,000 | 1.3654 | 1.3654 |
| 24,000 | 1.3413 | 1.3381 |

From step 12,000 to step 24,000, validation loss fell from 1.4377 to 1.3381. The final sample follows Mia and a key through several paragraphs. The story from step 12,000 stayed on the snow theme but repeated the snowman-building action and introduced a boy abruptly. Use those as concrete checks for later samples. [Saved generation](https://github.com/sampath017/MiniReasoner/blob/d2f04074bf97bcf310bc987a122a6316ec8c9677/notebooks/main.ipynb)

**Choose the base checkpoint.** Step 24,000 has the lowest logged validation loss. Compare it with step 12,000 as a reference and the runner-up at 23,500 if the choice is close. Rescore on the same held-out story windows, ideally the full validation split once. Generate from each on the same five prompts with greedy decoding and the same length cap. Look for repetition, consistent characters and a coherent continuation. Record the chosen step, validation loss, run ID and Azure checkpoint. Preserve the original checkpoint and tokenizer when starting SFT.

## 2 Define arithmetic and measure the baseline

Start with addition and subtraction over nonnegative one- and two-digit operands. Include zero, equal operands, carries and borrowing. Begin with nonnegative results. Track negative answers, longer operands, multiplication and short word problems separately as later challenges.

Build the answer key with integer arithmetic. Split by the underlying arithmetic problem before writing different phrasings. Keep `7 + 8` and `8 + 7` in the same split, as well as ‘What is seven plus eight?’ if it is used. Keep subtraction order meaningful. Reserve fresh training prompts for RL, a development set for choosing checkpoints, and a sealed final test set. Save the generator version, split seed and counts by difficulty.

Use the same question and answer format across comparisons. For example, prompt `What is 47 + 8?\nAnswer:` and expect `55` followed by EOS. Score one greedy completion per question. Record exact-answer accuracy, valid-format rate and correct/total by operation, operand length and carry or borrow. Run this once on the selected base model before any arithmetic updates. Retain the fixed story windows and five story prompts as a language baseline.

## 3 Teach arithmetic with supervised fine-tuning

Initialize from the selected base weights and vocabulary. Start a new optimizer and a separate MLflow run linked to the base checkpoint. Train on verified question–answer demonstrations and compute loss only on the answer and EOS. Keep each example within the 256-token context. Include both symbolic and plain-English questions.

Run a small pilot first. Inspect wrong answers such as copying an operand, missing a carry, or getting the number right but continuing to write. Track greedy exact-answer accuracy on the development set alongside supervised loss and story retention. Expand examples in weak categories, then repeat the comparison on the same development problems.

**Decision.** Keep the best SFT checkpoint as a usable model and as the reference for RL. If it frequently ignores the answer format or nearly always misses a category, improve the demonstrations first. If it already solves the target range reliably, evaluate a harder range before spending GPU time on RL.

## 4 Improve harder cases with verifiable rewards

Start from the selected SFT checkpoint. On training prompts, sample several answers per problem, for example four. Give reward **1** for a completion containing exactly one valid integer equal to the computed answer, followed by EOS or the agreed stop. Give **0** for an incorrect or malformed completion, conflicting numbers, or extra text. Check the verifier on zero, negative signs if enabled, multiple guesses and empty answers.

Run a small group relative policy optimization (GRPO) pilot on problems where the SFT model sometimes succeeds. A group containing both correct and wrong completions provides a learning signal. If almost all groups fail, simplify the problems or add supervised examples. If almost all succeed, increase the difficulty. Keep the SFT reference frozen and monitor answer length, invalid answers, the fraction of mixed groups and held-out greedy accuracy. [GRPO paper](https://arxiv.org/abs/2402.03300)

**Decision.** Compare the RL checkpoint with SFT using one greedy completion on the same development questions. Count questions RL fixed and questions it broke. Check each difficulty category and repeat the story evaluation. Promote RL only if the gain is larger than the uncertainty of the paired comparison and language quality stays within the tolerance chosen before the pilot. Keep SFT if it performs as well.

## 5 Final evaluation and release

Freeze the chosen candidate, prompt format, decoding settings and verifier. Run the sealed arithmetic test once for reporting. Report exact-answer accuracy with counts by operation and difficulty, invalid answers, and examples of carry, borrowing and wording failures. Report unfamiliar wordings and longer operands separately so it is clear what generalizes. Keep the published SimpleStories test reserved for the final language evaluation.

Keep the selected inference weights, model config and original tokenizer together. Verify that a known prompt produces the expected result after loading on CPU. Retain a full resumable checkpoint with optimizer state in Azure. Keep metrics and parent-run links in Databricks and the prepared story corpus in Drive. Record the files and checksums before considering any cleanup.

## Decisions and revisions

| Date or milestone | Evidence to record | Decision |
| --- | --- | --- |
| 24 Nov 2026 | Validation fell from 1.4377 at 12,000 to 1.4152 at 14,000 | Keep the 24,000-step cap and the 32M architecture |
| 24 Nov 2026 | Step 24,000 saved to Azure, mean training loss 1.3413 and validation loss 1.3381 | Pretraining complete. Compare checkpoints on matched story windows and prompts before SFT. |
| After arithmetic baseline | Per-category counts, accuracy and story scores | Set target accuracy and acceptable story regression before training |
| After SFT pilot | Base vs. SFT, errors by difficulty and retained story quality | Expand SFT, pilot RL, or keep SFT |
| After RL pilot | SFT vs. RL on identical questions, including regressions | Select release candidate |
| At release | Frozen test results, CPU load check and artifact locations | Record final model version |
