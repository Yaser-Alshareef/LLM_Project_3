import json
import os
import time
import random
import torch
import streamlit as st
import evaluate
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

load_dotenv()

# ── Page config ──
st.set_page_config(page_title="Model Comparison Demo", layout="wide")
st.title("DeepSeek vs LoRA TinyLlama — Live Demo")

# ── Load test data (cached) ──
@st.cache_data
def load_test_data():
    data = []
    with open(Path("data/clean_dolly_test.json"), "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

test_data = load_test_data()
categories = sorted(set(ex["category"] for ex in test_data))

# ── Load LoRA model (cached) ──
@st.cache_resource
def load_lora_model():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, "model/model")
    model.eval()
    return tokenizer, model

with st.spinner("Loading LoRA TinyLlama..."):
    tokenizer, lora_model = load_lora_model()

# ── DeepSeek client ──
client = OpenAI(
    api_key=os.getenv("DEEP_API"),
    base_url="https://api.deepseek.com"
)

# ── Model functions ──
def ask_deepseek(prompt):
    start = time.time()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Follow the instruction precisely and give a concise answer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.3
    )
    latency = time.time() - start
    output = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else None
    return output, latency, tokens

def ask_lora(prompt):
    start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(lora_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = lora_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    output = tokenizer.decode(out[0], skip_special_tokens=True)
    output = output[len(prompt):].strip()
    latency = time.time() - start
    return output, latency

def build_prompt(instruction, context=""):
    prompt = f"Instruction: {instruction}\n"
    if context:
        prompt += f"Context: {context}\n"
    prompt += "Answer:"
    return prompt

# ── Sidebar: input mode ──
st.sidebar.header("Input")
mode = st.sidebar.radio("Choose input mode:", ["Random test example", "Custom instruction"])

instruction = ""
context = ""
reference = ""

if mode == "Random test example":
    cat_filter = st.sidebar.selectbox("Filter by category (optional):", ["All"] + categories)
    if st.sidebar.button("Pick random example"):
        pool = test_data if cat_filter == "All" else [ex for ex in test_data if ex["category"] == cat_filter]
        ex = random.choice(pool)
        st.session_state["example"] = ex

    if "example" in st.session_state:
        ex = st.session_state["example"]
        instruction = ex["instruction"]
        context = ex.get("context", "")
        reference = ex["response"]
        st.sidebar.markdown(f"**Category:** {ex['category']}")
else:
    instruction = st.sidebar.text_area("Instruction:", height=100)
    context = st.sidebar.text_area("Context (optional):", height=80)

# ── Main area ──
if instruction:
    st.markdown("### Instruction")
    st.write(instruction)
    if context:
        st.markdown("### Context")
        st.write(context[:500] + "..." if len(context) > 500 else context)
    if reference:
        with st.expander("Reference Answer"):
            st.write(reference)

    prompt = build_prompt(instruction, context)

    if st.button("Run Both Models", type="primary"):
        col1, col2 = st.columns(2)

        # DeepSeek
        with col1:
            st.markdown("#### DeepSeek")
            with st.spinner("Calling DeepSeek API..."):
                ds_output, ds_latency, ds_tokens = ask_deepseek(prompt)
            st.success(f"Latency: {ds_latency:.2f}s | Tokens: {ds_tokens}")
            st.write(ds_output)

        # LoRA TinyLlama
        with col2:
            st.markdown("#### LoRA TinyLlama")
            with st.spinner("Running local inference..."):
                lora_output, lora_latency = ask_lora(prompt)
            st.success(f"Latency: {lora_latency:.2f}s | Local")
            st.write(lora_output)

        # Evaluation (only if reference exists)
        if reference:
            st.markdown("---")
            st.markdown("### Automatic Evaluation vs Reference")
            with st.spinner("Computing BERTScore..."):
                bertscore = evaluate.load("bertscore")
                ds_bs = bertscore.compute(
                    predictions=[ds_output], references=[reference], lang="en"
                )
                lora_bs = bertscore.compute(
                    predictions=[lora_output], references=[reference], lang="en"
                )

            eval_col1, eval_col2 = st.columns(2)
            with eval_col1:
                st.metric("DeepSeek BERTScore F1", f"{ds_bs['f1'][0]:.4f}")
            with eval_col2:
                st.metric("LoRA BERTScore F1", f"{lora_bs['f1'][0]:.4f}")
        else:
            st.info("No reference answer — showing outputs only (no automatic evaluation).")
else:
    st.info("Pick a test example or type a custom instruction in the sidebar to get started.")