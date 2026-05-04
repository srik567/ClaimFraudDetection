# Generation Prompt

The following prompt was used to generate this application.

---

## The "Production Fraud Architect"

> Act as a Senior AI Engineer and Insurance Fraud Architect. Build a production-grade Python system for Automated Duplicate Claim & Tampering Detection with a Human-in-the-Loop feedback mechanism.

### Core Architecture Requirements

1. **Data Schemas**: Use Pydantic to define `Claim` (invoice_id, amount, patient_name, hospital_id, timestamp) and `ReviewResult` models.

2. **Extraction Agent**: Implement a module that normalizes data (lowercase, stripping whitespace) and handles OCR confidence scores.

3. **Forensic Agent**:
   - **Metadata Scrutiny**: Detect if PDFs were saved by unauthorized software (Photoshop, Canva).
   - **Digital Tamper Check**: Implement an Error Level Analysis (ELA) function using Pillow to flag font overlays or altered amount fields.

4. **Auditor Agent (The Matcher)**:
   - Implement Exact Matching via SHA-256 hashing.
   - Implement Fuzzy Matching (using `thefuzz`) to detect slightly altered invoice numbers (e.g., `'101'` vs `'I01'`).
   - Implement Cross-Reference logic: Flag if (Patient + Date + Amount) matches an existing record even if the Invoice ID is different.

5. **Evaluation & Metrics Module**:
   - Use `scikit-learn` to calculate and print Accuracy, Precision, and Recall.
   - Generate a Confusion Matrix visual in the console.
   - Implement Manual Review Triggers: Automatically flag claims with a risk score between 40–75 for human verification.

6. **Active Learning Feedback Loop**:
   - Use a `sqlite3` database to store 'AI Risk Score' vs 'Human Final Decision'.
   - Create a `retrain_thresholds()` function: If humans consistently approve a specific 'flagged' pattern, the system should automatically lower its sensitivity for that pattern.

---

### Execution Logic

Provide a `main.py` that simulates 10 claims, including:
- 1 Perfect Duplicate.
- 1 Digital Tamper (Altered Total).
- 1 Fuzzy Match (Altered Invoice Number).
- 1 'Grey Area' claim for manual review.

Print a detailed **Error Analysis Report** at the end, highlighting exactly which features (Metadata, ELA, or Fuzzy Match) correctly identified the fraud.

---

### Precision Check

Look at the console output for "Precision Score". If it's below 0.9, the system is "over-flagging" legitimate users.

### Manual Review Flag

Check the list of claims marked `STATUS: PENDING_REVIEW`. The system should explain why it's unsure (e.g., `"Reason: High visual similarity but unique Invoice ID"`).

### Feedback Test

Run the script, "submit" a human override via the provided function, and run it again. You should see the Confusion Matrix improve.

---

### Mock Data Generator

A `mock_data/generator.py` script is included so you can test with 100+ claims immediately.

```bash
python3 -m mock_data.generator --count 200 > claims.json
```

---

### System Architecture Requirements

1. **Pydantic Data Schemas**: Define `Claim` and `ReviewResult` models.
2. **Extraction Agent (OCR Logic)**: Mock a module that extracts `invoice_id`, `hospital_id`, `amount`, `patient_name`, and `date`. Use `thefuzz` for normalization.
3. **Forensic Agent (Digital Tampering)**:
   - Implement Metadata analysis (checking for Photoshop/Canva signatures).
   - Implement a 'Structural Similarity' placeholder to detect color-photocopy layouts.
   - Implement 'Error Level Analysis' (ELA) logic using PIL to detect digital font overlays.
4. **Auditor Agent (Matching Engine)**:
   - Perform Exact Match (Hash-based).
   - Perform Fuzzy Match on Invoice IDs.
   - Perform Cross-Reference (Same Patient + Date + Amount).
5. **Evaluation & Metrics Module**:
   - Calculate Accuracy, Precision, and Recall.
   - Generate a Confusion Matrix (True Positives, False Positives, etc.).
   - Function to identify 'Grey-Area' claims (Risk Score 40–75) for manual review.
6. **Feedback Loop (Active Learning)**:
   - Implement a SQLite database to store 'AI Predictions' vs 'Human Decisions'.
   - Create a 'Retrain' function that adjusts matching thresholds based on Human overrides (e.g., if humans keep approving 85% fuzzy matches, auto-adjust threshold to 90%).

---

### Code Specifications

- Use `sqlite3` for the feedback database.
- Use `scikit-learn` for metrics and confusion matrix.
- Ensure the code is modular, well-commented, and includes a 'Main' block to simulate 10 claims with intentional duplicates and tampered files.
- Print a final **Error Analysis Report** explaining why certain claims failed.
