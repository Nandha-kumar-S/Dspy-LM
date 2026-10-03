# Prompt Optimisation with DSPy

Optimising prompts *programmatically* instead of by hand — using
[DSPy](https://github.com/stanfordnlp/dspy) to compile a small local model
(Llama-3.2-3B-Instruct, served with vLLM) into a better classifier for a
specialist domain.

Applied to two different tasks to check the approach transfers:

| Task | Type | Where |
|---|---|---|
| **Astronautics MCQA** | multiple-choice question answering | `main.py` + `scripts/` + `config/` |
| **Forest health classification** | 4-class classification from 18 ecological measurements | `code.ipynb` |

---

## The problem

A small model given a zero-shot prompt does badly on specialist domains. The
usual response is to hand-tune the prompt — reword the instruction, add an
example, try again. That's slow, unrepeatable, and there's no way to tell
whether a change actually helped or you just got lucky on the examples you
happened to eyeball.

DSPy reframes prompting as a **compilation** problem: you declare the
input/output signature, and an optimiser searches for the prompt and
demonstrations that maximise a metric you define.

## Approach

**1. Declare the signature — what goes in, what comes out**

```python
class Astronautics_QA(dspy.Signature):
    """Answer the question using astronautics knowledge"""
    question = dspy.InputField(desc="question based on astronautics")
    choices  = dspy.InputField(desc="list of multiple choices for the question")
    answer   = dspy.OutputField(desc="should best a matching answer from choices")
```

No prompt string anywhere — the signature is the specification, and DSPy
generates the actual prompt.

**2. Wrap it in a reasoning module**

```python
class Classification(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought(Astronautics_QA)
```

`ChainOfThought` makes the model produce intermediate reasoning before its
answer, rather than answering in one shot.

**3. Measure a baseline before optimising**

This step is the point of the whole exercise. Without a zero-shot number
there's no way to claim the optimiser did anything:

```python
evaluator = dspy.Evaluate(devset=test_set, metric=answer_exact_match,
                          num_threads=8, display_progress=True)
eval_result = evaluator(classifier)
```

**4. Compile**

```python
teleprompter_fsrs = BootstrapFewShotWithRandomSearch(
    metric=answer_exact_match,
    max_labeled_demos=16,
    max_bootstrapped_demos=2,
    num_candidate_programs=8,
    max_rounds=5,
)
optimised_classifier = teleprompter_fsrs.compile(classifier, trainset=train_set)
```

`BootstrapFewShotWithRandomSearch` runs the model over the training set, keeps
the traces where it got the answer right, and uses those as few-shot
demonstrations — then random-searches over candidate combinations, scoring
each against the metric. The saved checkpoint shows it settled on **16
demonstrations**.

**5. Re-evaluate and save**

The compiled program — instructions plus selected demonstrations — is saved to
JSON so it can be reloaded without re-running the (expensive) search.

## Results

Forest health classification, zero-shot baseline:

```
Average Metric: 32 / 200 (16.0%)
```

Worth noting what that means: it's a **4-class** problem, so random guessing
scores ~25%. The un-optimised 3B model is performing **below chance** — it
isn't weakly right, it's systematically wrong. That's exactly the situation
few-shot demonstrations are meant to fix, and it makes a clean "before".

**The post-optimisation score isn't recorded in this repo** — the evaluation
cell was run but its output wasn't saved with the notebook. The compiled
programs in `saved_program/` are the artifacts that survived.

## Layout

```
main.py                      astronautics pipeline, end to end
config/
  model_config.yaml          model name, API base, keys
  data_config.yaml           data dir, input/output columns, save path
scripts/
  predictor.py               DSPy Signature + Module
  utils.py                   dataframe prep, prompt-history inspection
data/
  astronautics_mcqa.csv      source dataset (226 rows)
  train.csv / test.csv       141 / 60 split
  column_label.json          field descriptions for the forest-health task
saved_program/               compiled programs (JSON)
code.ipynb                   forest health task, exploratory
dataset_analysis.ipynb       dataset inspection
```

## Running it

**1. Serve the model**

```bash
vllm serve unsloth/Llama-3.2-3B-Instruct --dtype float16 --api-key any_key
```

**2. Point the config at it** — `config/model_config.yaml`:

```yaml
model_name: 'openai/unsloth/Llama-3.2-3B-Instruct'
api_base: "http://0.0.0.0:8000/v1"
api_key: "abc123"      # must match the --api-key above; vLLM accepts anything
```

**3. Run**

```bash
pip install -r requirements.txt
python main.py
```

It prints a prediction before optimisation, the baseline score, the optimised
score, a prediction after, and writes the compiled program to
`saved_program/program_checkpoint.json`.

## Notes and limitations

**The dataset is small.** 141 training examples for astronautics. With
`num_candidate_programs=8` and `max_rounds=5`, there's real scope for the
random search to overfit the 60-example test set — it's being used both to
select the program and to report the score. A separate held-out set would make
the final number trustworthy.

**`answer_exact_match` is strict.** The whole pipeline lowercases everything
(`process_df`) precisely because 'Mars' vs 'mars' would otherwise be scored as
wrong. Any other formatting drift — trailing punctuation, a restated option
letter — still counts as a miss, so the metric likely understates real accuracy.

**Two tasks share one repo.** `saved_program/` holds artifacts from both; the
forest-health work lives only in the notebook and was never moved into the
`main.py` structure.

**`program_checkpoint_8b.json`** is from an 8B model run that isn't otherwise
reflected in the configs.

## What I'd change

- **Record the post-optimisation numbers.** The baseline is captured but the
  payoff isn't, which undercuts the whole story.
- **Hold out a third split** for final reporting, so the test set isn't doing
  double duty as both selector and scorer.
- **Try `MIPROv2`** — it optimises the instruction text as well as the
  demonstrations, where `BootstrapFewShotWithRandomSearch` only selects
  examples.
- **A softer metric than exact match** — normalise punctuation and option
  prefixes before comparing, so formatting noise doesn't read as a wrong answer.
- **Move the forest-health task into the `main.py` structure** so both tasks run
  the same way.

## Built with

DSPy · Llama-3.2-3B-Instruct · vLLM · pandas · scikit-learn
