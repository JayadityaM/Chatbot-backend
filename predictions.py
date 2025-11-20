from src.groq_api import ask_groq
import json, time, numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load dataset
with open("nitr_testset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

MODELS = [
    ("LLaMA 3.1 8B Instant", "llama-3.1-8b-instant"),
    ("LLaMA 3.1 70B Versatile", "llama-3.1-70b-versatile"),
    ("Gemma 2 9B", "gemma2-9b-it"),
    ("Mixtral 8x7B", "mixtral-8x7b")
]

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
smooth = SmoothingFunction().method1

results_summary = []

for model_name, model_id in MODELS:
    print(f"\n🚀 Evaluating {model_name} ...")
    bleu_scores, rouge_scores, times = [], [], []

    for sample in tqdm(data, desc=model_name):
        instruction = sample["instruction"]
        context = sample.get("context", "")
        reference = sample["expected_output"]

        if sample["task_type"] == "summarization":
            question = f"{instruction}\n\n{context}"
        else:
            question = instruction

        prompt = f"{instruction}\n\n{context}"

        start_time = time.time()
        try:
            model_output = ask_groq(prompt, question, model_name=model_id)
        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue
        end_time = time.time()

        bleu = sentence_bleu([reference.split()], model_output.split(), smoothing_function=smooth)
        rouge = scorer.score(reference, model_output)

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        times.append(end_time - start_time)

    # Aggregate results
    avg_bleu = np.mean(bleu_scores)
    avg_rouge1 = np.mean([r["rouge1"].fmeasure for r in rouge_scores])
    avg_rougeL = np.mean([r["rougeL"].fmeasure for r in rouge_scores])
    avg_time = np.mean(times)

    results_summary.append({
        "model": model_name,
        "BLEU": avg_bleu,
        "ROUGE-1": avg_rouge1,
        "ROUGE-L": avg_rougeL,
        "Avg Time (s)": avg_time
    })

# Print Results
print("\n===== MODEL COMPARISON RESULTS =====")
for res in results_summary:
    print(f"\n🧠 {res['model']}")
    print(f"   BLEU: {res['BLEU']:.4f}")
    print(f"   ROUGE-1: {res['ROUGE-1']:.4f}")
    print(f"   ROUGE-L: {res['ROUGE-L']:.4f}")
    print(f"   Avg Time(s): {res['Avg Time (s)']:.2f}")
