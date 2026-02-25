import json
import pandas as pd


def fill_prompt(template: str, context: dict) -> str:
    return template.format_map(context)


def call_llm(prompt_text: str, client, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return response.choices[0].message.content


def run_row(row: dict, prompts: list[dict], client, model: str) -> dict:
    context = dict(row)
    for prompt in prompts:
        filled = fill_prompt(prompt["template"], context)
        output = call_llm(filled, client, model)
        context[prompt["output_name"]] = output
    return context


def run_dataset(df: pd.DataFrame, prompts: list[dict], client, model: str) -> pd.DataFrame:
    output_names = [p["output_name"] for p in prompts]
    rows = []

    for _, row in df.iterrows():
        try:
            result = run_row(row.to_dict(), prompts, client, model)
        except Exception as e:
            result = row.to_dict()
            for name in output_names:
                result[name] = f"ERROR: {e}"
        rows.append(result)

    result_df = pd.DataFrame(rows)

    # JSON expansion: flatten dict outputs into dot-notation columns
    for name in output_names:
        if name not in result_df.columns:
            continue
        expanded = {}
        any_dict = False
        for val in result_df[name]:
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, dict):
                        any_dict = True
                        for k, v in parsed.items():
                            expanded.setdefault(k, []).append(v)
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
            for k in expanded:
                expanded[k].append(None)

        if any_dict:
            for k, values in expanded.items():
                result_df[f"{name}.{k}"] = values

    return result_df
