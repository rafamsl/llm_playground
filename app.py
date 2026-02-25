import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from llm_engine import run_row

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini"

st.set_page_config(page_title="LLM Playground", layout="wide")
st.title("LLM Playground")
st.caption("Test AI prompts against a real dataset — no engineering required. Upload your data, define what you want the AI to do, and see results in seconds.")

# ── Session state init ────────────────────────────────────────────────────────
if "prompts" not in st.session_state:
    st.session_state.prompts = []
if "result_df" not in st.session_state:
    st.session_state.result_df = None

# ── Section 1: Dataset ────────────────────────────────────────────────────────
st.header("1. Dataset")
st.caption("Upload the CSV you want to run the AI against. Each row will be processed individually. The column names will be available as variables in your prompts.")
uploaded = st.file_uploader("Upload CSV", type="csv")

df = None
columns = []
if uploaded:
    df = pd.read_csv(uploaded)
    columns = list(df.columns)
    st.dataframe(df.head(), use_container_width=True)

# ── Section 2: Prompt Chain ───────────────────────────────────────────────────
st.header("2. Prompt Chain")
st.caption("Write one or more prompts to run in sequence. Use `{column_name}` to inject values from your dataset. Each prompt's output becomes a variable you can reference in the next prompt — perfect for multi-step reasoning.")

if columns:
    st.caption(f"Available columns: {', '.join(f'`{{{c}}}`' for c in columns)}")
else:
    st.caption("Upload a CSV above to see available column names.")

for i, prompt in enumerate(st.session_state.prompts):
    # Seed the textarea key from prompts list (only on first render)
    tkey = f"template_{i}"
    if tkey not in st.session_state:
        st.session_state[tkey] = prompt["template"]

    with st.expander(f"Prompt {i + 1} → `{prompt['output_name']}`", expanded=True):
        col_tmpl, col_name = st.columns([4, 1])
        with col_tmpl:
            st.text_area(
                "Template",
                key=tkey,
                height=120,
                label_visibility="collapsed",
                placeholder="e.g. Summarize this message: {message}",
            )
            # Sync widget value back to prompts list
            st.session_state.prompts[i]["template"] = st.session_state[tkey]

            # Add placeholder
            all_vars = columns + [
                p["output_name"] for j, p in enumerate(st.session_state.prompts) if j < i
            ]
            if all_vars:
                ph_col, btn_col = st.columns([3, 1])
                with ph_col:
                    selected_var = st.selectbox(
                        "placeholder",
                        options=all_vars,
                        key=f"selectbox_{i}",
                        label_visibility="collapsed",
                    )
                with btn_col:
                    if st.button("+ Add placeholder", key=f"chip_{i}", use_container_width=True):
                        st.session_state[tkey] += f"{{{selected_var}}}"
                        st.rerun()
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

# ── Section 3: Run ────────────────────────────────────────────────────────────
st.header("3. Run")
st.caption("When you're ready, hit Run. The AI will process every row and add a new column for each prompt output. If an output is structured JSON, it will be automatically split into separate columns.")

if df is not None and st.session_state.prompts:
    if st.button("Run", type="primary"):
        progress = st.progress(0, text="Running…")
        result_rows = []
        output_names = [p["output_name"] for p in st.session_state.prompts]
        total = len(df)

        for idx, row in enumerate(df.itertuples(index=False)):
            row_dict = row._asdict()
            try:
                result = run_row(row_dict, st.session_state.prompts, client, MODEL)
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

elif df is None:
    st.info("Upload a CSV to get started.")
elif not st.session_state.prompts:
    st.info("Add at least one prompt to run.")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result_df is not None:
    st.header("Results")
    st.dataframe(st.session_state.result_df, use_container_width=True)

    csv_bytes = st.session_state.result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="results.csv",
        mime="text/csv",
    )
