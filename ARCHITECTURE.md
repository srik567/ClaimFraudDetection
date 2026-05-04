# ClaimFraudDetection Architecture Overview

## System Architecture Diagram

```mermaid
graph TB
    subgraph Input["Input Layer"]
        rawInput["Raw Claim Input<br/>(PDF, Image, OCR)"]
    end

    subgraph Processing["Processing Layer"]
        extractionAgent["Extraction Agent<br/>(mock_ocr_extract<br/>+ normalize_claim)"]
        claimSchema["Claim Schema<br/>(Pydantic Validation)"]
        forensicAgent["Forensic Agent"]
        auditorAgent["Auditor Agent"]
    end

    subgraph Analysis["Analysis Layer"]
        subgraph Forensic["Forensic Analysis"]
            metadataCheck["Metadata Scrutiny<br/>(Producer + Creator)"]
            elaCheck["Error Level Analysis<br/>(Pillow Recompression)"]
        end

        subgraph Audit["Audit Analysis"]
            exactMatch["Exact Match<br/>(SHA-256)"]
            fuzzyMatch["Fuzzy Match<br/>(thefuzz Invoice)"]
            crossReference["Cross-Reference<br/>(Patient+Date+Amount)"]
        end

        riskScorer["Risk Scorer<br/>(Weighted Feature Scoring)"]
    end

    subgraph Decision["Decision Layer"]
        reviewResult["ReviewResult Schema<br/>(risk_score, status,<br/>flags, reason)"]
        approved["APPROVED<br/>(risk < 40)"]
        pendingReview["PENDING_REVIEW<br/>(40 ≤ risk ≤ 75)"]
        flagged["FLAGGED<br/>(risk > 75)"]
    end

    subgraph Feedback["Feedback & Retraining"]
        humanReviewer["Human Reviewer<br/>(submit_human_override)"]
        feedbackDb["SQLite Feedback DB<br/>(AI score + human decision)"]
        retrainThresholds["retrain_thresholds<br/>(Adjust Pattern Sensitivity)"]
    end

    subgraph Evaluation["Evaluation Module"]
        metricsModule["Metrics Calculation<br/>(Accuracy, Precision, Recall)"]
        errorReport["Error Analysis Report<br/>(Confusion Matrix<br/>+ Feature Report)"]
    end

    %% Input to Processing
    rawInput --> extractionAgent
    extractionAgent --> claimSchema

    %% Processing to Analysis
    claimSchema --> forensicAgent
    claimSchema --> auditorAgent

    %% Forensic Analysis
    forensicAgent --> metadataCheck
    forensicAgent --> elaCheck
    metadataCheck --> riskScorer
    elaCheck --> riskScorer

    %% Audit Analysis
    auditorAgent --> exactMatch
    auditorAgent --> fuzzyMatch
    auditorAgent --> crossReference
    exactMatch --> riskScorer
    fuzzyMatch --> riskScorer
    crossReference --> riskScorer

    %% Decision Logic
    riskScorer --> reviewResult
    reviewResult --> approved
    reviewResult --> pendingReview
    reviewResult --> flagged

    %% Feedback Loop
    pendingReview --> humanReviewer
    flagged --> humanReviewer
    reviewResult --> feedbackDb
    humanReviewer --> feedbackDb
    feedbackDb --> retrainThresholds
    retrainThresholds --> forensicAgent
    retrainThresholds --> auditorAgent

    %% Evaluation
    approved --> metricsModule
    pendingReview --> metricsModule
    flagged --> metricsModule
    metricsModule --> errorReport

    style Input fill:#e1f5ff
    style Processing fill:#f3e5f5
    style Analysis fill:#fce4ec
    style Decision fill:#fff3e0
    style Feedback fill:#e8f5e9
    style Evaluation fill:#f1f8e9
```

## Architecture Layers

### 1. **Input Layer**
- Accepts raw claim data in multiple formats (PDF, images, OCR payloads)

### 2. **Processing Layer**
- **Extraction Agent**: Normalizes and extracts data from raw inputs
- **Claim Schema**: Validates extracted data using Pydantic models

### 3. **Analysis Layer**
- **Forensic Analysis**: Examines PDF metadata and image integrity
  - Metadata Scrutiny: Checks producer and creator fields
  - Error Level Analysis: Detects image manipulation via Pillow recompression
  
- **Audit Analysis**: Cross-references and deduplication checks
  - Exact Match: SHA-256 fingerprinting for identical claims
  - Fuzzy Match: Invoice similarity matching via thefuzz
  - Cross-Reference: Validates patient, date, and amount correlations
  
- **Risk Scorer**: Combines all signals with weighted feature scoring

### 4. **Decision Layer**
- Generates `ReviewResult` with risk score, status, flags, and reasoning
- Routes claims to one of three categories:
  - **APPROVED**: Risk score < 40
  - **PENDING_REVIEW**: Risk score between 40–75
  - **FLAGGED**: Risk score > 75

### 5. **Feedback & Retraining Layer**
- **Human Reviewer**: Reviews pending and flagged claims
- **SQLite Feedback DB**: Stores AI scores and human decisions
- **Retraining**: Adjusts forensic and audit thresholds based on human feedback

### 6. **Evaluation Module**
- Calculates performance metrics (Accuracy, Precision, Recall)
- Generates confusion matrix and feature-level error analysis
- Produces comprehensive error reports

## Data Flow Summary

1. Raw claim data is normalized and validated into a `Claim` model
2. Forensic and Auditor agents analyze the claim in parallel
3. All feature signals feed the risk scorer
4. Risk score determines the claim status and routing
5. Human decisions refine thresholds over time
6. Evaluation module tracks system performance
