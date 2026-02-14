# ICU Patient Outcome Prediction Using Time-Series Clinical Data
## Comprehensive Literature Review (2018-2026)

---

## 1. Landmark Studies and Papers

### 1.1 Foundational Benchmarks

**Harutyunyan et al. (2019) -- "Multitask Learning and Benchmarking with Clinical Time Series Data"**
- **Authors:** Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, Greg Ver Steeg, Aram Galstyan
- **Publication:** Scientific Data, Vol. 6, No. 1 (2019)
- **Key Contribution:** Proposed four standardized clinical prediction benchmarks using MIMIC-III: (1) in-hospital mortality, (2) physiologic decompensation, (3) length of stay (LOS), and (4) phenotype classification. Established a multi-task learning framework with channel-wise LSTMs and deep supervision. This became the de facto benchmark for clinical time-series ML research.
- **Results:** LSTM-based models significantly outperformed linear baselines. Deep supervision improved decompensation and LOS predictions. The benchmark covers 40,000+ ICU stays with rich multivariate time-series data.
- **Code:** https://github.com/YerevaNN/mimic3-benchmarks

**PhysioNet/Computing in Cardiology Challenge 2012 -- ICU Mortality Prediction**
- **Dataset:** 12,000 adult patients from MIMIC-II; 5 general descriptors + 36 time-series variables (vital signs and lab results) from first 48 hours of ICU stay.
- **Participation:** 37 teams, ~200 submissions worldwide.
- **Significance:** Remains a foundational benchmark (P12 dataset) for evaluating mortality predictors.

**PhysioNet/Computing in Cardiology Challenge 2019 -- Early Prediction of Sepsis**
- **Dataset:** 60,000+ ICU patients with up to 40 clinical variables per hour of ICU stay; sourced from three hospital systems.
- **Participation:** 104 groups, 853 submissions.
- **Key Finding:** Diverse computational approaches predict sepsis onset hours before clinical recognition, but generalizability across hospital systems remains a major challenge.
- **Evaluation:** Novel clinical utility-based metric rewarding early predictions and penalizing late/missed predictions and false alarms.

### 1.2 Google's Landmark EHR Study

**Rajkomar et al. (2018) -- "Scalable and Accurate Deep Learning with Electronic Health Records"**
- **Authors:** Alvin Rajkomar, Eyal Oren, Kai Chen, Andrew M. Dai et al. (Google, UCSF, Stanford, University of Chicago)
- **Publication:** npj Digital Medicine, Vol. 1, Article 18 (2018)
- **Data:** 216,221 adult patients from two US academic medical centers; 46.8 billion data points in FHIR format.
- **Results:**
  - In-hospital mortality: AUROC 0.93-0.94
  - 30-day unplanned readmission: AUROC 0.75-0.76
  - Prolonged length of stay: AUROC 0.85-0.86
  - Discharge diagnoses: frequency-weighted AUROC 0.90
- **Key Innovation:** Used raw, uncurated EHR data in FHIR format without site-specific harmonization. Demonstrated that deep learning can bypass labor-intensive feature engineering while outperforming traditional clinical prediction models.

### 1.3 Dynamic and Real-Time Prediction Studies

**Dynamic Survival Prediction from Uncurated Data (Scientific Reports, 2020)**
- Used all uncurated chart, lab, and output events from MIMIC-III without variable selection.
- 21,000+ ICU patients; strongly predictive recordings in the first few hours of stay.
- **Result:** AUROC 0.85 (95% CI 0.83-0.86) after 48 hours, outperforming SAPS II and OASIS within 12 hours of admission.

**TBAL -- Time-Aware Bidirectional Attention LSTM (JMIR, 2025)**
- **Data:** 176,344 ICU stays from MIMIC-IV and eICU-CRD; dynamic variables updated hourly.
- **Results:**
  - 12h-1d mortality on MIMIC-IV: AUROC 0.959 (95% CI 0.942-0.975)
  - 12h-1d mortality on eICU-CRD: AUROC 0.933 (95% CI 0.915-0.953)
  - Full-stay AUROC: 0.936 (MIMIC-IV), 0.919 (eICU-CRD)

**TECO -- Transformer-Based Encounter-Level Clinical Outcome Model (JAMIA Open, 2025)**
- COVID-19 cohort: AUC 0.89-0.97 across time intervals.
- MIMIC testing: AUC 0.65-0.77, outperforming RF (0.59-0.75) and XGBoost (0.59-0.74).

**RealMIP Framework (npj Digital Medicine, 2025)**
- Real-time mortality prediction using generative models for dynamic imputation of missing data.
- **Results:** AUC 0.957 internal, 0.968 on MIMIC-IV, 0.932 on SICdb.

**Real-Time Ensemble ML Model (Critical Care, 2024)**
- Ensemble of deep learning + LightGBM from South Korean academic institution.
- **External validation:** AUROC 0.746 (MIMIC), 0.798 (eICU-CRD), 0.819 (AmsterdamUMCdb).
- Outperformed NEWS, SPTTS, MEWS, APACHE-II, SAPS, and SOFA scores.

### 1.4 MIMIC-Sepsis Benchmark (arXiv, 2025)

- Curated benchmark for sepsis trajectories with tasks: mortality prediction, LOS estimation, shock classification.
- Transformer-based architectures consistently outperformed linear baselines.
- Key finding: integrating treatment variables substantially improves dynamic prediction (e.g., LSTM saw +0.112 absolute AUROC increase for vasopressor requirement task when treatment features were added).

### 1.5 CRISP Framework

**CRISP -- Causal Relationship Informed Superior Prediction**
- Integrates a causal graph into a Transformer-based architecture.
- **Results:** AUROC 0.948 on MIMIC-III and 0.917 on MIMIC-IV for in-hospital mortality using first 48 hours of data.

---

## 2. Notable Models and Systems

### 2.1 Google Health / DeepMind

**Rajkomar et al. (2018)** remains the most prominent Google contribution to ICU prediction. Key characteristics:
- Used FHIR-formatted raw EHR data across multiple institutions.
- Deep learning models (RNNs with attention) achieved AUROC 0.93-0.94 for in-hospital mortality.
- Demonstrated attribution-based interpretability for clinical transparency.
- Showed that the entire patient chart, including clinical notes, can be leveraged without manual feature selection.

### 2.2 InSight by Dascena

**System:** InSight -- ML-based sepsis prediction algorithm
- **Developer:** Dascena (Hayward, California)
- **FDA Status:** FDA-cleared for clinical use
- **Mechanism:** Gradient tree boosting using only six vital signs (systolic BP, pulse pressure, heart rate, respiration rate, temperature, SpO2) + age and GCS. Requires only 2 hours of preceding data.
- **Performance:** First sepsis screening system to exceed AUROC 0.90 using only vital sign inputs. Outperforms SIRS and SOFA in identifying/predicting sepsis, severe sepsis, and septic shock.
- **Clinical Impact:** Prospective trial showed earlier sepsis detection and reduced ICU length of stay.
- **Strengths:** Robust to missing data; customizable to new sites via transfer learning; integrates seamlessly into existing clinical workflows.

**Note:** In April 2024, the Prenosis Sepsis ImmunoScore received FDA de novo clearance -- the first FDA-authorized AI tool for predicting sepsis within 24 hours of ICU admission, marking a milestone for AI transitioning from research to routine clinical practice.

### 2.3 APACHE / SOFA / NEWS vs. Machine Learning

**Traditional Scoring Systems:**
- **APACHE II/IV:** Acute Physiology and Chronic Health Evaluation; uses 12 physiologic variables, age, and chronic health conditions. Typical AUROC: 0.80-0.85.
- **SOFA:** Sequential Organ Failure Assessment; assesses six organ systems. Typical AUROC: 0.70-0.80.
- **NEWS:** National Early Warning Score; uses 7 physiological parameters. Designed for general ward deterioration detection.
- **SAPS II:** Simplified Acute Physiology Score; 17 variables. Typical AUROC: ~0.81.

**ML vs. Traditional Scores -- Key Comparisons:**

| Study | ML Model | ML AUROC | Traditional Score | Trad. AUROC |
|-------|----------|----------|-------------------|-------------|
| Scientific Reports (2022) | Random Forest | 0.945 | APACHE II | ~0.85 |
| Gradient Boosting (MIMIC-III) | GBM | 0.927 | SAPS II | 0.809 |
| XMI-ICU (2025) | XGBoost | 0.920 | APACHE IV | 0.737 (+18.3%) |
| Real-Time Ensemble (2024) | DL + LGBM | 0.866 | NEWS | Lower (p<0.001) |
| Sepsis ML (Turkey, 2022) | Ensemble ML | ~85% accuracy | APACHE II/SAPS II | ~84% accuracy |

**Key Findings:**
- ML models generally outperform traditional scores (AUROC >0.90 vs 0.70-0.85).
- Hybrid approaches using ML to enhance traditional scores (e.g., LSTM over APACHE II/SOFA/SAPS II time series) show promise.
- Traditional scores remain valuable for simplicity, interpretability, and universal familiarity.
- A 2025 study in Internal and Emergency Medicine demonstrated that LSTM models capturing full temporal dynamics from APACHE II, SOFA, and SAPS II within the first 24 hours outperform the static scoring systems.

### 2.4 Work from Taiwan / National Taiwan University Hospital (NTUH)

**NTUH AI-Enabled ECG Research:**
- **CAD Prediction (Biomedicines, 2022):** Multi-center retrospective study (2018-2020) using AI to predict and localize coronary artery disease from 12-lead ECGs. 2,303 patients with angiography-proven CAD, 12,954 ECG records.
- **VPC Detection:** Deep learning CNN for ventricular premature contraction prediction during sinus rhythm.
- **Anesthesia Depth Prediction:** Deep learning (AlexNet, VGG19) using 512 Hz ECG and 128 Hz PPG signals.
- **Aortic Regurgitation Detection:** DL-based ECG models to predict LV remodeling associated with hemodynamically significant AR. Multi-center study with NTUH data (2008-2022).
- **Emergency Triage (JMIR, 2024):** Interpretable deep learning for triage level, hospitalization, and LOS prediction. Achieved 63% (triage), 82% (hospitalization), 71% (LOS) accuracy.

**Taipei Medical University ICU Work:**
- Topic model + gradient boosting for ICU mortality prediction using MIMIC data.
- Best model: Gradient Boosting achieved AUROC 0.928, specificity 93.16%, sensitivity 83.25%.

---

## 3. State-of-the-Art Architectures

### 3.1 Recurrent Neural Networks (LSTM / GRU)

**Standard LSTM/GRU:**
- The workhorse of clinical time-series prediction since ~2016.
- Channel-wise LSTMs (separate LSTM per variable) with deep supervision established as strong baselines by Harutyunyan et al. (2019).
- Typical mortality prediction AUROC: 0.85-0.87 on MIMIC-III benchmarks.

**GRU-D (Che et al., 2018) -- "Recurrent Neural Networks for Multivariate Time Series with Missing Values"**
- **Key Innovation:** Incorporates missing data patterns (masking + time intervals) directly into GRU architecture via trainable decay mechanisms.
- Instead of imputing then predicting, GRU-D treats missingness as informative and decays hidden states toward empirical means between observations.
- **Publication:** Scientific Reports, 2018; one of the most cited papers in clinical time-series ML.
- Outperformed non-RNN baselines significantly on MIMIC-III and PhysioNet.

**VS-GRU (Variable Sensitive GRU, Applied Sciences, 2019)**
- Extends GRU-D by learning features of different variables separately, accounting for variable-specific missing rates (often >80% in clinical data).
- Reduces harmful impact of high-missing-rate variables.

**Time-Aware LSTM Variants:**
- Extend the forget gate to logarithmic or cubic decay functions of time intervals.
- AMITA-LSTM (Adaptive Multi-scale Integrative Time-Aware LSTM) designed for irregularly collected sequential data.

### 3.2 Temporal Convolutional Networks (TCNs)

**Architecture:**
- Based on causal convolutions (output at time t depends only on inputs at time t and earlier).
- Dilated convolutions (inspired by WaveNet) enable exponentially growing receptive fields.
- Residual connections for deep networks.

**Key Studies:**
- **Attention-based TCN (BMC Anesthesiology, 2022):** AUROC 0.837 (48h mortality on MIMIC-III). TCN with attention mechanism for interpretability.
- **TCN for LOS and Mortality (Scientific Reports, 2022):** Evaluated on 23,944 patients from MIMIC-III using MIMIC-Extract pipeline with 24h time-series data.
- **TCN-FFNN (PMC, 2020):** Showed TCNs are less task-specific than LSTMs after hyperparameter tuning, making them more easily extensible to other clinical applications.

**Advantages over RNNs:**
- Highly parallelizable (no sequential dependency during training).
- Flexible receptive field with exponential scaling.
- Lower memory requirements during training.
- Efficient enough for real-time deployment on CPU-only systems.
- Bai et al. reported TCN performance matching or exceeding LSTM/GRU on many sequence tasks.

### 3.3 Transformer-Based Models

**STraTS -- Self-Supervised Transformer for Sparse and Irregularly Sampled Clinical Time-Series**
- **Authors:** Sindhu Tipirneni, Chandan K. Reddy
- **Publication:** ACM Transactions on Knowledge Discovery from Data, Vol. 16, No. 6 (2022)
- **Key Innovation:** Treats time-series as a set of observation triplets (time, variable, value) instead of dense matrices. Uses Continuous Value Embedding for encoding without discretization. Self-supervised pretraining via time-series forecasting as auxiliary task.
- **Results:** Outperformed GRU-D, IP-Net, SeFT on MIMIC-III mortality prediction. More robust to noise; generalizes well with limited labeled data. Interpretable variant (I-STraTS) identifies important measurements.

**Raindrop -- Graph-Guided Network for Irregularly Sampled Time Series**
- **Authors:** Xiang Zhang, Marko Zeman, Theodoros Tsiligkaridis, Marinka Zitnik (Harvard)
- **Publication:** ICLR 2022
- **Key Innovation:** Graph neural network using message passing between sensors. Each sample represented as a separate sensor graph with time-varying dependencies.
- **Results:** Outperformed SOTA by up to 11.4% absolute F1-score on PhysioNet P19 (38,803 patients) and P12 (11,988 patients).

**mTAND -- Multi-Time Attention Networks (ICLR 2021)**
- **Authors:** Satya Narayan Shukla, Benjamin M. Marlin
- Learns embedding of continuous-time values with attention mechanism producing fixed-length representations from variable-length observations.
- Outperformed IP-Net, SeFT, and RNN baselines on interpolation and classification tasks.

**TGAM -- Transformer with Temporal and Missing Value Handling**
- AUROC: 87.65% (MIMIC-III), 87.00% (MIMIC-IV), outperforming baselines by >5 percentage points.

**CRISP -- Causal Transformer (2024)**
- Integrates causal graphs into Transformer architecture.
- AUROC: 0.948 (MIMIC-III), 0.917 (MIMIC-IV) for in-hospital mortality.

**Other Notable Transformer Works:**
- **DuETT:** Attends over both time and event type dimensions; outperforms SOTA on MIMIC-IV and PhysioNet-2012.
- **ViTiMM:** Vision Transformer for irregular multi-modal measurements; outperforms SOTA on MIMIC-IV mortality and phenotyping (6,175 patients).
- **ContiFormer (NeurIPS 2023):** Continuous-time Transformer for irregular time series.
- **Time Series as Images (NeurIPS 2023):** Vision Transformer approach for irregularly sampled time series.

### 3.4 Neural ODE-Based Models

**GRU-ODE-Bayes (NeurIPS 2019)**
- **Authors:** Edward De Brouwer, Jaak Simm, Adam Arany, Yves Moreau
- Continuous-time GRU with Bayesian update network for sporadic observations.
- Can exactly represent Fokker-Planck dynamics of complex SDE-driven processes.
- Evaluated on healthcare and climate data.

**Latent ODE (NeurIPS 2019)**
- **Authors:** Yulia Rubanova et al.
- Variational autoencoder with ODE-based dynamics in latent space.
- Naturally handles arbitrary time gaps; models observation times via Poisson processes.
- Evaluated on PhysioNet 2012 ICU data.

**Neural Controlled Differential Equations (NeurIPS 2020)**
- Extended Neural ODEs for irregular time series.
- Evaluated on PhysioNet 2019 sepsis data (40,335 time series).
- Produced best AUC when observational intensity was used as a feature.

### 3.5 Handling Irregular Sampling and Missing Data

| Strategy | Representative Models | Mechanism |
|----------|----------------------|-----------|
| **Decay mechanisms** | GRU-D, Time-Aware LSTM | Modify RNN gates with time-decay functions |
| **Masking + time intervals** | GRU-D, VS-GRU | Encode missingness as explicit model inputs |
| **Observation triplets** | STraTS, SeFT | Treat each (time, variable, value) as a set element |
| **Graph message passing** | Raindrop | Model inter-sensor dependencies with GNN |
| **Continuous-time ODEs** | GRU-ODE-Bayes, Latent ODE, Neural CDE | Define dynamics in continuous time |
| **Attention over time** | mTAND, Transformers | Attend to arbitrary time points without fixed grid |
| **Self-supervision** | STraTS | Pretrain on unlabeled data with forecasting proxy |
| **Generative imputation** | RealMIP | Dynamic imputation via generative models |

### 3.6 Multi-Task Learning Approaches

**Joint Prediction of Multiple ICU Outcomes:**

- **Harutyunyan et al. (2019):** Heterogeneous MTL with channel-wise LSTMs predicting mortality, decompensation, LOS, and phenotype simultaneously. Deep supervision improved performance, especially for decompensation and LOS.

- **Explainable Time-Series DL (Frontiers in Medicine, 2022):** RNN, GRU, LSTM with attention for simultaneous prediction on MIMIC-IV (40,083 patients):
  - Mortality: AUC 0.870
  - Prolonged LOS: AUC 0.765
  - 30-day readmission: AUC 0.635

- **Multi-Task CNN from Clinical Notes (PMC, 2019):** Multi-level CNN trained jointly on mortality and LOS from MIMIC-III clinical notes. Multi-task models slightly outperformed single-task counterparts.

- **Concurrent Mortality + LOS (medRxiv, 2025):** Compared single-task XGBoost, multi-class XGBoost, and MLP-based MTL. Task-specific XGBoost: AUC 0.92 (mortality), 0.83 (LOS quartiles for elective admissions).

- **M3T-LM (Computers in Biology and Medicine, 2024):** Multi-modal multi-task learning model jointly predicting LOS and mortality.

---

## 4. Key Benchmarks, Performance, and Challenges

### 4.1 Summary of AUROC Results

#### In-Hospital Mortality Prediction (General ICU)

| Model / Approach | Dataset | Observation Window | AUROC |
|------------------|---------|--------------------|-------|
| TBAL (Dynamic LSTM) | MIMIC-IV | 12h-1d | 0.959 |
| RealMIP | MIMIC-IV | Real-time | 0.968 |
| CRISP (Causal Transformer) | MIMIC-III | 48h | 0.948 |
| CRISP (Causal Transformer) | MIMIC-IV | 48h | 0.917 |
| Gradient Boosting | MIMIC-III | 24h | 0.927 |
| Rajkomar et al. (Google) | Two US hospitals | Full stay | 0.93-0.94 |
| XMI-ICU (MI patients) | eICU | 6h | 0.920 |
| Random Forest | MIMIC-III | 24h | 0.945 |
| TGAM (Transformer) | MIMIC-III | -- | 0.877 |
| Attention-based TCN | MIMIC-III | 48h | 0.837 |
| RNN/GRU/LSTM + Attention | MIMIC-IV | 24h | 0.870 |
| Gradient Boosting (Taiwan) | MIMIC | -- | 0.928 |
| Dynamic Survival (uncurated) | MIMIC-III | 48h | 0.850 |
| Hybrid CNN-BiLSTM (vitals only) | ICU data | -- | 0.884 |

#### External Validation Performance (Cross-Dataset)

| Model | Internal AUROC | External AUROC | External Dataset |
|-------|----------------|----------------|------------------|
| XGBoost (SA-AKI) | 0.878 (MIMIC-IV) | 0.720 | eICU |
| Real-Time Ensemble | 0.866 (internal) | 0.746-0.819 | MIMIC/eICU/Amsterdam |
| TBAL | 0.959 (MIMIC-IV) | 0.933 | eICU-CRD |

### 4.2 Most Important Predictive Features

**Vital Signs (ranked by importance):**
1. **Heart rate** and **heart rate variability/complexity** -- among the strongest vital sign predictors
2. **Respiratory rate** -- consistently top-2 predictor alongside heart rate
3. **Blood pressure** (systolic and diastolic) -- highest Chi-squared association with deterioration
4. **SpO2** (peripheral oxygen saturation)
5. **Temperature**
6. **Glasgow Coma Scale (GCS)** -- top predictor across mortality, LOS, and readmission models

**Laboratory Values:**
1. **Blood urea nitrogen (BUN)** -- co-selected across mortality, LOS, and readmission models
2. **Lactate** -- key acid-base imbalance predictor
3. **Anion gap**
4. **Platelet count**
5. **Total bilirubin**
6. **Bicarbonate / base excess / pH** -- acid-base balance indicators
7. **PTT (Partial Thromboplastin Time)**

**Demographics:**
- **Age** -- universally important
- **Ethnicity** -- noted for readmission prediction

**Treatment Variables:**
- **Norepinephrine** (vasopressors) -- significant for mortality
- **Invasive ventilation** -- significant for prolonged LOS
- Including treatment variables substantially improves dynamic predictions (e.g., +0.112 AUROC for vasopressor requirement in MIMIC-Sepsis benchmark)

### 4.3 Key Datasets Used in the Field

| Dataset | Source | Size | Key Features |
|---------|--------|------|--------------|
| **MIMIC-III** | Beth Israel Deaconess, Boston | ~60,000 ICU stays | 17 vital signs, labs, notes, waveforms |
| **MIMIC-IV** | Same as above (updated) | ~70,000+ ICU stays | Updated through 2022 |
| **eICU-CRD** | Philips multi-center | 200,000+ ICU stays | 208 hospitals across US |
| **PhysioNet P12** | MIMIC-II subset | 12,000 patients | 36 time-series, 48h window |
| **PhysioNet P19** | Multi-hospital | 38,803 patients | 34 sensors, sepsis labels |
| **AmsterdamUMCdb** | Amsterdam UMC | ~20,000 admissions | European ICU data |
| **SICdb** | Salzburg Univ Hospital | -- | European validation set |

### 4.4 Major Challenges

**1. Irregular Sampling and Missing Data**
- Clinical time series contain >80% missing values in some variables.
- Sampling rates vary between variables (vitals hourly, labs every 4-12h).
- Missingness is often "informative" -- correlated with patient severity.
- Standard imputation introduces bias and strong assumptions.

**2. Generalizability and External Validation**
- Performance consistently drops on external datasets (e.g., 0.878 to 0.720 AUROC).
- Population differences, practice patterns, and documentation standards vary between institutions.
- Approximately half of published ICU mortality prediction models have NOT been externally validated.
- The YAIB (Yet Another ICU Benchmark) found that MIMIC-IV and eICU transfer reasonably well to each other, but cross-hospital generalization remains difficult.

**3. Data Heterogeneity**
- ICU data encompasses demographics, vitals, labs, waveforms, clinical notes, imaging, treatments.
- Heterogeneous coding schemes across institutions.
- Multiple data modalities require different preprocessing and modeling strategies.

**4. Dynamic Patient Conditions**
- Patient states change rapidly with interventions.
- Most research focuses on first 24-48 hours of admission, but clinical utility requires continuous real-time prediction.
- Accuracy of static window predictions is lower than dynamic models that update continuously.

**5. Clinical Deployment Barriers**
- Explainability remains a major concern for clinical adoption (SHAP, attention weights are common but imperfect).
- Laborious manual data input for some scoring systems.
- Lab processing delays and workflow constraints limit real-time feature availability.
- Regulatory approval pathway is complex (only a few FDA-cleared systems exist).
- Prospective validation studies remain rare.

**6. Class Imbalance**
- ICU mortality rates are typically 10-15%, creating significant class imbalance.
- Data rebalancing techniques (oversampling, undersampling, SMOTE) are necessary but can introduce artifacts.

**7. Ethical and Privacy Concerns**
- MIMIC and eICU require data use agreements; privacy constraints limit data sharing.
- Algorithmic bias across demographic groups is an active concern.
- Need for federated learning and privacy-preserving approaches.

### 4.5 Emerging Directions (2024-2026)

1. **Causal inference integration** -- Moving beyond correlational predictions to identify genuine risk factors (e.g., CRISP framework).
2. **Federated learning** -- Training across institutions without sharing patient data.
3. **Foundation models for clinical time series** -- Pre-trained models adaptable to downstream tasks.
4. **Organ-specific outcome prediction** -- Beyond binary mortality to AKI, ventilator weaning success, neurological outcomes.
5. **Multi-modal fusion** -- Combining structured time-series, clinical notes, imaging, and waveform data.
6. **Latent SDE models** -- Probabilistic forecasting with uncertainty quantification for treatment effect estimation.
7. **Standardized preprocessing pipelines** -- Addressing reproducibility issues across studies.

---

## References and Sources

### Landmark Papers
- Harutyunyan et al. "Multitask Learning and Benchmarking with Clinical Time Series Data." Scientific Data 6, 96 (2019). https://www.nature.com/articles/s41597-019-0103-9
- Rajkomar et al. "Scalable and Accurate Deep Learning with Electronic Health Records." npj Digital Medicine 1, 18 (2018). https://www.nature.com/articles/s41746-018-0029-1
- Che et al. "Recurrent Neural Networks for Multivariate Time Series with Missing Values." Scientific Reports (2018). https://www.nature.com/articles/s41598-018-24271-9

### Transformer and Attention Models
- Tipirneni & Reddy. "Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series." ACM TKDD 16(6), 2022. https://dl.acm.org/doi/10.1145/3516367
- Zhang et al. "Graph-Guided Network for Irregularly Sampled Multivariate Time Series" (Raindrop). ICLR 2022. https://arxiv.org/abs/2110.05357
- Shukla & Marlin. "Multi-Time Attention Networks for Irregularly Sampled Time Series." ICLR 2021. https://arxiv.org/abs/2101.10318

### Neural ODE Models
- De Brouwer et al. "GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series." NeurIPS 2019. https://arxiv.org/abs/1905.12374
- Rubanova et al. "Latent ODEs for Irregularly-Sampled Time Series." NeurIPS 2019. https://arxiv.org/abs/1907.03907
- Kidger et al. "Neural Controlled Differential Equations for Irregular Time Series." NeurIPS 2020.

### TCN and CNN Models
- "Learning to Predict In-Hospital Mortality Risk in the ICU with Attention-Based TCN." BMC Anesthesiology (2022). https://link.springer.com/article/10.1186/s12871-022-01625-5
- "Temporal Convolutional Networks and Data Rebalancing for Clinical LOS and Mortality Prediction." Scientific Reports (2022). https://www.nature.com/articles/s41598-022-25472-z
- "Temporal Convolutional Networks Allow Early Prediction of Events in Critical Care." PMC (2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7647248/

### Benchmarks and Comparisons
- PhysioNet Challenge 2019. "Early Prediction of Sepsis from Clinical Data." https://pmc.ncbi.nlm.nih.gov/articles/PMC6964870/
- PhysioNet Challenge 2012. "Predicting In-Hospital Mortality of ICU Patients." https://pmc.ncbi.nlm.nih.gov/articles/PMC3965265/
- "Yet Another ICU Benchmark (YAIB)." https://openreview.net/pdf?id=ox2ATRM90I
- MIMIC-Sepsis Benchmark. https://arxiv.org/html/2510.24500v1

### Clinical Systems and Scoring
- Dascena InSight. https://ai.nejm.org/doi/full/10.1056/AIoa2400867
- "Comparison of Severity of Illness Scores and AI Models for ICU Mortality." PMC (2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9198821/
- "Prediction Algorithm for ICU Mortality and LOS Using ML." Scientific Reports (2022). https://www.nature.com/articles/s41598-022-17091-5
- "Enhancing Mortality Prediction: Improving APACHE II, SOFA, and SAPS II Using LSTM." Internal and Emergency Medicine (2025). https://link.springer.com/article/10.1007/s11739-025-03896-5

### Real-Time and Dynamic Models
- "Real-Time ML Model to Predict Short-Term Mortality in Critically Ill Patients." Critical Care (2024). https://ccforum.biomedcentral.com/articles/10.1186/s13054-024-04866-7
- "Dynamic Real-Time Risk Prediction Model for ICU Patients (TBAL)." JMIR (2025). https://www.jmir.org/2025/1/e69293
- "Unlocking the Potential of Real-Time ICU Mortality Prediction (RealMIP)." npj Digital Medicine (2025). https://www.nature.com/articles/s41746-025-02114-y

### Multi-Task and Explainable Models
- "Explainable Time-Series Deep Learning Models for ICU." Frontiers in Medicine (2022). https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2022.933037/full
- "Explainable ML for ICU Mortality in MI Patients (XMI-ICU)." Scientific Reports (2025). https://www.nature.com/articles/s41598-025-13299-3
- "Concurrent Prediction of Mortality and LOS Using Multi-Task ML." medRxiv (2025). https://www.medrxiv.org/content/10.1101/2025.03.21.25324386v1.full

### Taiwan-Based Research
- "AI-Enabled ECG Algorithm for CAD Prediction at NTUH." Biomedicines (2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC8962407/
- "Predicting ICU Mortality by Topic Model (Taipei Medical University)." PubMed (2022). https://pubmed.ncbi.nlm.nih.gov/35742138/
- "Interpretable Deep Learning for Triage (NTUH)." JMIR (2024). https://pubmed.ncbi.nlm.nih.gov/38557661/

### Surveys and Reviews
- "Leveraging MIMIC Datasets for Better Digital Health." arXiv (2025). https://arxiv.org/html/2506.12808v1
- "AI-Based Models for ICU Mortality Prediction: A Scoping Review." JICM (2024). https://journals.sagepub.com/doi/10.1177/08850666241277134
- "Prognostic Performance of AI/ML for ICU Mortality: Systematic Review." Cureus (2025). https://www.cureus.com/articles/365925
- "A Review of Irregular Time Series Data Handling with Gated RNNs." Neurocomputing (2021). https://www.sciencedirect.com/science/article/abs/pii/S0925231221003003
