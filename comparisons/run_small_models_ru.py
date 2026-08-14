import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "привет",
    "Как тебя зовут?",
    "Какая столица Франции?",
    "Напиши короткое стихотворение о море.",
    "Почему нам нужен сон?",
    "Сколько будет 2 плюс 2?",
    "Объясни просто, как работает компьютер.",
    "Расскажи о своём дне.",
    "Чем ты любишь заниматься для развлечения?",
    "Можешь помочь спланировать поездку?",
    "Какая сегодня погода?",
    "Опиши свою любимую еду.",
    "Что ты думаешь о музыке?",
    "Как приготовить бутерброд?",
    "Расскажи короткую историю.",
    "Сколько будет 17 умножить на 3?",
    "Что такое фотосинтез?",
    "Дай совет, как выучить новый язык.",
    "Какая разница между котом и собакой?",
    "Напиши короткий диалог между двумя друзьями.",
    "Что такое гравитация?",
    "Посоветуй книгу для чтения.",
    "Как справиться со стрессом?",
    "Что произойдёт, если нагреть лёд?",
    "Напиши хайку про осень.",
    "Объясни разницу между Python и JavaScript.",
    "Какой твой любимый цвет и почему?",
    "Реши: если поезд едет 60 км/ч, сколько он проедет за 2.5 часа?",
]

MODELS = [
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "smollm2_135m"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5_0.5b"),
]

for model_id, tag in MODELS:
    print(f"\n{'='*20} {model_id} {'='*20}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s, params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    results = []
    for i, prompt in enumerate(PROMPTS):
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
        print(f"[{i+1}/{len(PROMPTS)}] ({dt:.1f}s) {prompt}\n  -> {response}\n", flush=True)
        results.append({"prompt": prompt, "response": response, "gen_time_s": dt})

    with open(f"dev-ignore/smallmodel_{tag}_responses.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    del model, tokenizer
