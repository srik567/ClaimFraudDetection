# ClaimFraudCheck Data Flow Diagram

```mermaid
flowchart TD
    rawInput["Raw Claim Input\nPDF, image, OCR payload"] --> extractionAgent["Extraction Agent\nmock_ocr_extract + normalize_claim"]

    extractionAgent --> claimSchema["Claim Schema\nPydantic validation"]
    claimSchema --> forensicAgent["Forensic Agent"]
    claimSchema --> auditorAgent["Auditor Agent"]

    forensicAgent --> metadataCheck["Metadata Scrutiny\nProducer + Creator fields"]
    forensicAgent --> elaCheck["Error Level Analysis\nPillow recompression diff"]

    metadataCheck --> forensicFlags["Forensic Flags\nMetadata, ELA"]
    elaCheck --> forensicFlags

    auditorAgent --> exactMatch["Exact Match\nSHA-256 fingerprint"]
    auditorAgent --> fuzzyMatch["Fuzzy Match\nthefuzz invoice similarity"]
    auditorAgent --> crossReference["Cross-Reference\nPatient + Date + Amount"]

    exactMatch --> auditorFlags["Auditor Flags\nExact, Fuzzy, CrossRef"]
    fuzzyMatch --> auditorFlags
    crossReference --> auditorFlags

    forensicFlags --> riskScorer["Risk Scorer\nWeighted feature scoring"]
    auditorFlags --> riskScorer

    riskScorer --> reviewResult["ReviewResult Schema\nrisk_score, status, flags, reason"]

    reviewResult --> approved["APPROVED\nrisk < 40"]
    reviewResult --> pendingReview["PENDING_REVIEW\n40 <= risk <= 75"]
    reviewResult --> flagged["FLAGGED\nrisk > 75"]

    approved --> metricsModule["Evaluation Module\nAccuracy, Precision, Recall"]
    pendingReview --> metricsModule
    flagged --> metricsModule

    pendingReview --> humanReviewer["Human Reviewer\nsubmit_human_override"]
    flagged --> humanReviewer

    reviewResult --> feedbackDb["SQLite Feedback DB\nAI score + human decision"]
    humanReviewer --> feedbackDb

    feedbackDb --> retrainThresholds["retrain_thresholds\nAdjust pattern sensitivity"]
    retrainThresholds --> auditorAgent
    retrainThresholds --> forensicAgent

    metricsModule --> errorReport["Error Analysis Report\nConfusion Matrix + Feature Report"]
```

## Flow Summary

1. Raw claim data is normalized and validated into a `Claim` model.
2. The Forensic Agent checks PDF metadata and runs ELA image analysis.
3. The Auditor Agent checks exact duplicates, fuzzy invoice matches, and patient/date/amount cross-references.
4. All feature signals feed the risk scorer, which creates a `ReviewResult`.
5. Claims are routed to `APPROVED`, `PENDING_REVIEW`, or `FLAGGED`.
6. Human decisions are stored in SQLite and used by `retrain_thresholds()` to adjust sensitivity over time.
7. The Evaluation module prints metrics, a confusion matrix, and feature-level error analysis.
