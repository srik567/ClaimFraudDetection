# ClaimFraudCheck — Automated Duplicate Claim & Tampering Detection

A production-grade Python system for insurance fraud detection with a Human-in-the-Loop (HITL) active-learning feedback mechanism.

---

## Features

| Component | Capability |
|-----------|-----------|
| **Extraction Agent** | Normalises claim fields, mocks OCR confidence scoring |
| **Forensic Agent** | PDF metadata analysis (Photoshop/Canva detection) + Error Level Analysis (ELA) via Pillow |
| **Auditor Agent** | SHA-256 exact duplicate hashing, fuzzy invoice matching (`thefuzz`), cross-reference matching |
| **Evaluation Module** | Accuracy, Precision, Recall, ASCII Confusion Matrix (`scikit-learn`) |
| **Feedback Loop** | SQLite-backed human override store + `retrain_thresholds()` active learning |
| **Mock Data Generator** | Generate 100+ synthetic claims with injected fraud patterns via CLI |

---

## Project Structure

```
ClaimFraudCheck/
├── requirements.txt
├── main.py                      # Run 10-claim simulation + Error Analysis Report
├── schemas/
│   └── models.py                # Pydantic: Claim, ReviewResult, FraudDecision
├── agents/
│   ├── extraction_agent.py      # Normalisation + OCR confidence mock
│   ├── forensic_agent.py        # PDF metadata check + ELA
│   └── auditor_agent.py         # Exact / Fuzzy / Cross-reference matching
├── evaluation/
│   └── metrics.py               # sklearn metrics + confusion matrix
├── feedback/
│   └── feedback_loop.py         # SQLite store + retrain_thresholds()
└── mock_data/
    └── generator.py             # 100+ claim generator + CLI
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the 10-claim simulation
python main.py

# 3. Generate 200 mock claims (writes JSON to stdout)
python -m mock_data.generator --count 200

# 4. Submit a human override and see thresholds update
python -c "
from feedback.feedback_loop import FeedbackLoop
fb = FeedbackLoop()
fb.submit_human_override('INV-002', 'APPROVED', notes='Legitimate re-submission')
fb.retrain_thresholds()
"

# 5. Re-run to see improved Confusion Matrix
python main.py
```

---

## Risk Score Logic

| Score Range | Decision |
|-------------|----------|
| > 75 | FLAGGED |
| 40 – 75 | PENDING_REVIEW (human queue) |
| < 40 | APPROVED |

| Signal | Score Added |
|--------|------------|
| Exact hash duplicate | +100 |
| Cross-reference match | +60 |
| Fuzzy invoice match | +50 |
| Metadata tamper flag | +30 |
| ELA tamper flag | +20 |

---

## Feedback Loop

The SQLite database (`fraud_feedback.db`) stores every AI prediction alongside the human reviewer's final decision.  
Running `retrain_thresholds()` inspects the override history: if humans consistently approve claims flagged by a specific pattern (≥70% approval rate), the matching threshold for that pattern is automatically raised by 5 points, reducing false positives on future runs.

---

## Metrics Output (example)

```
============================================================
           FRAUD DETECTION — ERROR ANALYSIS REPORT
============================================================

[SKLEARN METRICS]
  Accuracy  : 0.90
  Precision : 0.92
  Recall    : 0.88

[CONFUSION MATRIX]
              Predicted
              NEG    POS
  Actual NEG |  6  |  0 |
         POS |  1  |  3 |

[FEATURE REPORT]
  Metadata Check   : 1 TP, 0 FP
  ELA Analysis     : 1 TP, 1 FP
  Exact Hash Match : 1 TP, 0 FP
  Fuzzy Match      : 1 TP, 0 FP
  Cross-Reference  : 0 TP, 0 FP

[PENDING_REVIEW QUEUE]
  INV-005 | risk=55 | Reason: High visual similarity but unique Invoice ID
```
