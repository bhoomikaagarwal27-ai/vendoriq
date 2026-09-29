# VendorIQ v2: AI vendor analytics for any dataset

End-term project for AI Applications, use case #6: vendor selection / procurement recommender (App format).

Upload **one or many** vendor files in **any format and any column layout**. The app then works through these steps:

1. **Reads and combines** the files. CSV, TSV and TXT (any delimiter or encoding), Excel (every sheet, title rows skipped),
   JSON and Parquet are supported. Several files are **stacked** if they have the same columns, or **joined** on a shared key.
2. **Profiles** every column. It infers the kind of column (ID, name, category, number, Yes/No, Low/Medium/High, date)
   and understands messy values such as `₹ 1,250`, `2.5 %`, `7 days`, `1.2 lakh`, `12,5` and `10-12`.
3. **Gemini explores** the profile and proposes an analysis plan: the ID and name columns, the grouping,
   the criteria, whether higher or lower is better, weights, filters and derived metrics.
4. **A second Gemini call judges** that plan and corrects it ("LLM-as-a-judge").
5. **Python validates** the plan against the real data. Anything invalid is repaired and logged.
6. **You edit** any criterion, weight, direction or limit. Limits accept any value.
7. **Cleans and ranks** the data: implausible values are flagged, gaps are filled or excluded, and duplicates are removed.
   A transparent weighted score is calculated, followed by a 500-run stability test, the Pareto set and a rule-of-thumb (L1) check.
8. **Further analysis**: trade-offs and synergies between criteria, a correlation heatmap and group comparison.
9. **Gemini explains** the results. Every answer is checked (IDs, numbers, format) before it is shown. There is also a grounded chat.

Without an API key, a keyword-and-statistics plan and rule-based text keep the app fully working.
All sample data is fictional.

## Files

```
app.py          Streamlit interface (6 tabs)
ingest.py       read any file type, detect header/delimiter/encoding, stack or join files
profiler.py     parse messy numbers, map Yes/No and Low/Med/High, infer column kinds, privacy-safe profile
planner.py      keyword plan, plan validator (final authority), safe derived formulas, cleaning, filters
scoring.py      weighted scoring, stability test, Pareto front, rule-of-thumb check, decision label
analysis.py     trade-offs, correlation matrix, group summary, context for the AI, offline insights
ai_engine.py    Gemini calls (explorer, judge, analyst, chat), model fallback, output checks
prompts.py      the four system prompts, versioned
safety.py       prompt-injection detection
samples.py      demo datasets: chemical vendors, 2 messy packaging files, large synthetic, edge cases
sample_data/    vendors_sample.csv, vendors_edge_cases.csv, vendor_template.csv
tests/          24 automated tests (no internet needed)
.streamlit/config.toml   theme + 200 MB upload limit
requirements.txt
```

## Updating an app that is already deployed (from v1)

1. On GitHub, open your repository and go to **Add file → Upload files**.
2. Drag in **all the `.py` files** from the new zip, plus `requirements.txt` and `README.md`. Files with the same name are
   replaced; new files (`ingest.py`, `profiler.py`, `planner.py`, `analysis.py`, `safety.py`, `samples.py`) are added.
   Then click **Commit changes**.
3. Open `.streamlit/config.toml` on GitHub, click ✏️, and change `maxUploadSize = 2` to `maxUploadSize = 200`. Commit.
4. Optional: drag in the `tests` folder, and delete the old `validation.py` (open the file → ⋯ → **Delete file**). It is no longer used.
5. Streamlit reinstalls the requirements and restarts by itself (2–4 minutes). If it doesn't, go to **Manage app → ⋮ → Reboot app**.

## First-time deployment

1. Get a free Gemini API key at **aistudio.google.com/apikey**.
2. Upload every file and folder to a public GitHub repository. Drag the folders themselves so the folder structure is kept.
3. Go to **share.streamlit.io**, choose **Create app → Deploy a public app from GitHub**, and set the main file to `app.py`.
   Under **Advanced settings → Secrets**, enter `GEMINI_API_KEY = "your key"`. Then click Deploy.

## Limits

- Up to 20 files per upload and 200 MB per file.
- Up to 500,000 rows are analysed. Larger data is sampled.
- The AI only ever receives a statistical profile and the top-ranked rows, so AI cost does not grow with file size.
- Free Streamlit apps sleep after 12 hours without visitors. Anyone can wake the app with one click.
