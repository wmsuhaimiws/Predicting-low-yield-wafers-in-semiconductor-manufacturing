# Publishing this project to GitHub

A 5-minute checklist to get the repo live and rendering cleanly for graders.

## 0. One-time: confirm the repo root

Make sure you are in the `secom-yield-optimization/` folder (the one containing
`README.md`). Everything below is run from there.

```bash
cd secom-yield-optimization
```

## 1. Initialise git and make the first commit

```bash
git init
git add .
git status            # sanity check: no raw data, no .ipynb_checkpoints (see .gitignore)
git commit -m "SECOM yield optimization: end-to-end data-management project"
```

> `.gitignore` already excludes the raw datasets and caches, so they will not be
> committed — the data is fetched at runtime instead.

## 2. Create the GitHub repo and push

**Option A — GitHub CLI (fastest):**

```bash
gh repo create secom-yield-optimization --public --source=. --remote=origin --push
```

**Option B — web UI:** create an empty repo named `secom-yield-optimization` on
github.com (no README/license — this repo already has them), then:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/secom-yield-optimization.git
git push -u origin main
```

## 3. Verify it renders (what a grader sees)

- **README.md** — opens as the repo landing page; badges, tables and the repo map
  render. Section-1 industry sources are clickable.
- **notebooks/secom_yield_analysis.ipynb** — GitHub renders notebooks inline. If you
  want the *outputs* (charts, tables) visible without anyone running it, commit an
  executed copy: in Colab do `File -> Download -> .ipynb` after a clean
  `Restart session -> Run all`, replace the file, then commit:
  ```bash
  git add notebooks/secom_yield_analysis.ipynb
  git commit -m "Add executed notebook with outputs"
  git push
  ```
- **docs/pipeline_diagram.svg** — renders as an image on GitHub.
- **docs/*.md** — all the write-ups (methodology, insights, recommendations,
  executive summary, big-data) render as formatted pages.

## 4. (Optional) Commit rendered figures

Figures are generated, not committed by default. To include them so they show on
GitHub:

```bash
# locally (not Colab):
pip install -r requirements.txt
python -m src.run_pipeline          # writes PNGs into figures/ and the dashboard HTML
git add figures/*.png docs/interactive_dashboard.html
git commit -m "Add generated EDA figures and interactive dashboard"
git push
```

(Or download the figures from your Colab run and drop them into `figures/`.)

## 5. Final professional touches (already done / quick wins)

- [x] `LICENSE` (MIT, with a data-use notice)
- [x] `requirements.txt` pinned
- [x] `.gitignore`
- [x] `/data /notebooks /src /figures /docs` structure
- [ ] Add repo **Description** + **Topics** on GitHub: `semiconductor`,
      `yield-optimization`, `imbalanced-classification`, `iiot`, `secom`, `hive`,
      `pig`, `data-management` — improves discoverability and looks polished.
- [ ] Pin the repo on your profile for the interview.

## Suggested commit-history shape (looks deliberate, not dumped)

```
1. Initial repo scaffold + README + requirements
2. Data loading and cleaning pipeline
3. EDA figures and imbalance-aware modelling
4. Hive/Pig big-data component
5. Creativity extras: Pareto, dashboard, AI4I add-on
6. Docs: methodology, insights, recommendations, executive summary
7. Executed notebook with outputs
```

You can squash everything into one commit if you prefer a clean single drop —
either is fine for grading.
