# 📘 Project README

## 🚀 Project Overview
This repository contains:
- `streamlit_app.py` → Streamlit frontend for displaying results.
- `model.py` → ML model code (executed separately, results saved).
- `requirements.txt` → Dependencies.
- `results/` → Weekly outputs (CSV/JSON/plots).
- `dataset/` → Ignored in Git, local-only raw data.

---

## 🧩 Git Workflow

### Branching
- **main** → stable, display-ready Streamlit app.
- **weekX** → weekly progress branches (e.g., `week1`, `week2`).
- Work on weekly branches, merge into `main` when done.

Commands:
```bash
# Create new weekly branch
git checkout -b week2
git push origin week2

# Merge into main at end of week
git checkout main
git merge week2
git push origin main
```
---

### Tagging Milestones

Use tags to mark completed features or weekly summaries:
```bash
# Create tag
git tag [Week X]-<Feature/Task>-<Action>


# Push tag to remote
git push origin [Week X]-<Dates>

```

---

## 📝 Commit Message Style Guide

Format:

```bash
[Week X] <Feature/Task>: <Action>
```

Examples
```bash
RMSE: finalized calculation and validation
Model: optimized hyperparameters
Streamlit: added results dashboard
```
---

## 📒 Weekly Progress Log

Keep a CHANGELOG.md or update here:
```bash
## Week 1 (Jan 30 – Feb 5)
- Added baseline model
- Integrated Streamlit display
- Setup .gitignore

## Week 2 (Feb 6 – Feb 12)
- Finalized RMSE calculation
- Optimized model hyperparameters
- Updated results.json for Streamlit
```

---
## ✅ Best Practices
- Pull before coding: git pull origin branchname
- Commit small, meaningful changes (max 3/day is perfect).
- Push daily to sync across machines.
- Keep datasets out of Git (dataset/ ignored).
- Use Streamlit caching (st.cache_data) for faster reloads.
