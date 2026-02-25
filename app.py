import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from llm_engine import run_dataset

load_dotenv()
client = OpenAI()

st.set_page_config(page_title="LLM Playground", layout="wide")
st.title("LLM Playground")

# ── Session state init ────────────────────────────────────────────────────────
if "prompts" not in st.session_state:
    st.session_state.prompts = []
if "result_df" not in st.session_state:
    st.session_state.result_df = None

# ── Section 1: Task Setup ─────────────────────────────────────────────────────
st.header("1. Task Setup")
col1, col2 = st.columns([3, 1])
with col1:
    task_name = st.text_input("Task name", placeholder="e.g. Message classification")
with col2:
    model = st.text_input("Model", value="gpt-4o-mini")

# ── Section 2: Prompt Builder ─────────────────────────────────────────────────
st.header("2. Prompt Chain")
st.caption("Use `{column_name}` to reference CSV columns or previous prompt outputs.")

for i, prompt in enumerate(st.session_state.prompts):
    with st.expander(f"Prompt {i + 1} → `{prompt['output_name']}`", expanded=True):
        col_tmpl, col_name = st.columns([4, 1])
        with col_tmpl:
            st.session_state.prompts[i]["template"] = st.text_area(
                "Template",
                value=prompt["template"],
                key=f"template_{i}",
                height=120,
                label_visibility="collapsed",
                placeholder="e.g. Summarize this message: {message}",
            )
        with col_name:
            st.session_state.prompts[i]["output_name"] = st.text_input(
                "Output column",
                value=prompt["output_name"],
                key=f"output_name_{i}",
            )
        if st.button("Remove", key=f"remove_{i}"):
            st.session_state.prompts.pop(i)
            st.rerun()

if st.button("+ Add Prompt"):
    n = len(st.session_state.prompts)
    st.session_state.prompts.append({"template": "", "output_name": f"output_{n + 1}"})
    st.rerun()

# ── Section 3: Upload & Run ───────────────────────────────────────────────────
st.header("3. Dataset")
uploaded = st.file_uploader("Upload CSV", type="csv")

df = None
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

if df is not None and st.session_state.prompts:
    if st.button("Run", type="primary"):
        progress = st.progress(0, text="Running…")
        result_rows = []
        output_names = [p["output_name"] for p in st.session_state.prompts]
        total = len(df)

        from llm_engine import run_row
        import json

        for idx, row in enumerate(df.itertuples(index=False)):
            row_dict = row._asdict()
            try:
                result = run_row(row_dict, st.session_state.prompts, client, model)
            except Exception as e:
                result = dict(row_dict)
                for name in output_names:
                    result[name] = f"ERROR: {e}"
            result_rows.append(result)
            progress.progress((idx + 1) / total, text=f"Row {idx + 1} / {total}")

        result_df = pd.DataFrame(result_rows)

        # JSON expansion
        for name in output_names:
            if name not in result_df.columns:
                continue
            col_data = result_df[name].tolist()
            expanded = {}
            any_dict = False
            for val in col_data:
                parsed_dict = None
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, dict):
                            parsed_dict = parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
                if parsed_dict is not None:
                    any_dict = True
                    for k, v in parsed_dict.items():
                        expanded.setdefault(k, []).append(v)
                    for k in [k for k in expanded if k not in parsed_dict]:
                        expanded[k].append(None)
                else:
                    for k in expanded:
                        expanded[k].append(None)

            if any_dict:
                for k, values in expanded.items():
                    result_df[f"{name}.{k}"] = values

        st.session_state.result_df = result_df
        progress.empty()

elif df is not None and not st.session_state.prompts:
    st.info("Add at least one prompt to run.")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result_df is not None:
    st.header("Results")
    st.dataframe(st.session_state.result_df, use_container_width=True)

    csv_bytes = st.session_state.result_df.to_csv(index=False).encode("utf-8")
    filename = f"{task_name or 'results'}.csv"
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )
