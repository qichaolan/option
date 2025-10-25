## Quick orientation for code-writing agents

This repository implements small command-line tools for downloading stock data (Finviz Elite),
computing technical indicators, and generating AI-backed analyses using OpenAI.
Keep instructions concise and concrete — below are the essential conventions, workflows,
and examples that let an agent be immediately productive.

### Big-picture architecture
- `finviz.py` — HTTP client for Finviz Elite. Exposes `StockDownloader` which returns a CSV reader
  (or writes a file). It expects a YAML auth file containing `auth_token`.
- `stock_analyzer.py` + `stock_indicators.py` — data flow: CSV/DictReader -> pandas.DataFrame ->
  `StockIndicators` (calculates RSI/MFI/MACD/MA) -> `StockAnalyzer` (glues download + indicators).
- `openai_analyzer.py` — formats the latest indicator values and calls the OpenAI API
  to produce textual analyses. Prompts live in the `prompts/` folder.

Typical data flow: Finviz CSV -> DataFrame -> StockIndicators (adds columns like `RSI_14`,
`MACD`, `MA20`) -> `get_latest_indicators()` -> `OpenAIAnalyzer.format_indicator_data()` -> OpenAI.

### Contracts & quick examples (inputs / outputs)
- StockDownloader.download_stock_detail_data(ticker, days) -> csv.DictReader or None.
- StockAnalyzer.analyze_stock(ticker, days) -> dict with keys: `stock` (`StockIndicators`),
  `latest` (dict), `data` (DataFrame), `output_file` (CSV path).
- StockIndicators.get_latest_indicators() -> dict with date, close, volume, and indicator keys
  (note: some keys are lowercased when returned; see implementation for exact names).
- OpenAIAnalyzer.analyze_stock_indicators(ticker, latest_data, prompt_name) -> str (analysis) or None.

Examples (use these in tests or as templates):
- Run stock analysis (Finviz auth file default: `auth.yaml`):
  python stock_analyzer.py QQQ --days 365 --auth config/finviz_auth.yaml
- Run quick AI analysis (requires `openai_config.yaml` with `openai_api_key`):
  python openai_analyzer.py
  (or import `quick_analyze_with_ai()` in scripts/tests)

### Key files and conventions to reference
- Authentication/config
  - Finviz auth: YAML with key `auth_token` (example path: `config/finviz_auth.yaml`).
  - OpenAI config: modules expect `openai_config.yaml` by default (root) but a sample/alternate
    exists at `config/openai.yaml`. Do NOT hard-code keys; prefer env or a non-committed config file.

- Prompt templates: `prompts/default_analysis.txt`, `prompts/tech_analysis.txt` — load by name
  (pass `prompt_name='default_analysis'`), extension `.txt` is appended automatically.

- Indicator column names (exact spellings used across modules):
  `RSI_14`, `MFI_14`, `MACD`, `MACD_Signal`, `MACD_Histogram`, `MA20`, `MA50`, `MA200`.
  `StockIndicators.get_latest_indicators()` maps a subset of these to lowercased keys in the
  returned `latest` dict used by `OpenAIAnalyzer` (see source for exact mapping).

### Developer workflows & gotchas
- Dependencies: project uses `pandas`, `requests`, `pyyaml`, and `openai`. Install them before running.
- Finviz responses sometimes return HTML when the auth token is invalid — code checks for HTML and
  prints helpful diagnostics. If you see HTML in logs, verify `auth_token` and subscription status.
- OpenAI client initialization checks for the `openai` package and raises a clear ImportError
  with install instructions when missing.
- Many modules use `print()` for progress and debugging; prefer lightweight modifications that
  keep these messages or convert to logging consistently if you change behavior.

### What to change or extend (concrete pointers)
- Adding a new prompt: create `prompts/<name>.txt`. Use `OpenAIAnalyzer.list_available_prompts()`
  to verify it is discoverable.
- If you change the OpenAI config path, update `OpenAIAnalyzer.__init__()` default `config_file`
  parameter or pass the path explicitly when instantiating.
- When extending indicators, add column names to both `StockIndicators.get_latest_indicators()` and
  `OpenAIAnalyzer.format_indicator_data()` to ensure AI input is complete.

### Security and secrets
- Do not commit API keys. The repo contains a sample `config/openai.yaml` — treat it as an example.
  The project expects a file named `openai_config.yaml` in the root by default; add it to `.gitignore`.

### Minimal checklist for agents (before opening a PR)
1. Run `python stock_analyzer.py --help` and `python openai_analyzer.py` to validate CLI examples.
2. Confirm required config files exist (or mock them in tests): `config/finviz_auth.yaml` and
   `openai_config.yaml` (or pass alternate paths).
3. When you add indicator columns, update both `StockIndicators.get_latest_indicators()` and
   `OpenAIAnalyzer.format_indicator_data()` so the AI receives consistent inputs.

If anything above is unclear or you want the instructions expanded around tests, CI, or a
sample script for end-to-end runs, tell me which area to expand and I will update this file.
