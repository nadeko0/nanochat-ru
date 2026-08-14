import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS_EN = [
    "hi",
    "What is your name?",
    "What is the capital of France?",
    "Write a short poem about the sea.",
    "Why do we need sleep?",
    "What is 2 plus 2?",
    "Explain simply how a computer works.",
    "Tell me about your day.",
    "What do you like to do for fun?",
    "Can you help me plan a trip?",
    "What's the weather like today?",
    "Describe your favorite food.",
    "What do you think about music?",
    "How do you make a sandwich?",
    "Tell me a short story.",
    "What is 17 times 3?",
    "What is photosynthesis?",
    "Give me advice on learning a new language.",
    "What's the difference between a cat and a dog?",
    "Write a short dialogue between two friends.",
    "What is gravity?",
    "Recommend a book to read.",
    "How do I deal with stress?",
    "What happens if you heat ice?",
    "Write a haiku about autumn.",
    "Explain the difference between Python and JavaScript.",
    "What's your favorite color and why?",
    "Solve: if a train travels 60 km/h, how far does it go in 2.5 hours?",
]

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
print(f"\n{'='*20} {model_id} (English prompts) {'='*20}", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s, params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

results = []
for i, prompt in enumerate(PROMPTS_EN):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    dt = time.time() - t1
    print(f"[{i+1}/{len(PROMPTS_EN)}] ({dt:.1f}s) {prompt}\n  -> {response}\n", flush=True)
    results.append({"prompt": prompt, "response": response, "gen_time_s": dt})

with open("dev-ignore/smallmodel_smollm2_135m_english_responses.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("DONE")
