# Deploying AI Clinical Decision Support Systems in Taiwanese Hospitals
## Comprehensive Research Report

**Date:** 2026-02-17
**Focus:** Regulatory, technical, ethical, and operational requirements for deploying an AI early warning system in a Taiwanese hospital ICU, with emphasis on NTUH

---

## Table of Contents

1. [Taiwan FDA (TFDA) Regulatory Pathway for Clinical AI](#1-taiwan-fda-tfda-regulatory-pathway-for-clinical-ai)
2. [NTUH-Specific AI Deployment](#2-ntuh-specific-ai-deployment)
3. [Technical Requirements for Clinical Deployment](#3-technical-requirements-for-clinical-deployment)
4. [Ethical and IRB Considerations for Taiwan](#4-ethical-and-irb-considerations-for-taiwan)
5. [Examples of Successfully Deployed ICU AI Systems Worldwide](#5-examples-of-successfully-deployed-icu-ai-systems-worldwide)

---

## 1. Taiwan FDA (TFDA) Regulatory Pathway for Clinical AI

### 1.1 Regulatory Authority and Legal Framework

Taiwan's medical device regulation is overseen by the **Taiwan Food and Drug Administration (TFDA)** under the **Ministry of Health and Welfare (MOHW)**. The core legislation is the **Medical Devices Act** (effective May 1, 2021), which replaced the earlier pharmaceutical-centric framework and established a dedicated regulatory pathway for medical devices, including Software as a Medical Device (SaMD).

**Key regulatory documents for AI/ML medical devices:**
- **Medical Devices Act (2021)** — Core law governing all medical devices
- **Guidance for the Inspection and Registration of Medical Software of AI/ML-based Technologies (Sept 11, 2020)** — First AI/ML-specific guidance
- **Technical Guidelines for CADe/CADx AI/ML Medical Devices (Aug 14, 2025)** — Most current guidelines, replacing Sept 2023 version
- **Post-market Change Application Guidelines for SaMD (March 2022)** — Governs algorithm updates post-approval
- **Guidelines for Classification of Medical Software** — Determines whether software qualifies as a medical device

### 1.2 SaMD Classification System

The TFDA uses a **risk-based 3-tier classification** aligned with international (IMDRF) standards:

| Class | Risk Level | Description | Examples |
|-------|-----------|-------------|----------|
| **Class I** | Low | Software that transmits, stores, or displays medical data without modification | Basic X-ray image viewer, data storage software |
| **Class II** | Medium | Software with advanced capabilities that assists (but does not replace) clinical decision-making | CADe systems, medical image processing, surgical planning software, AI-assisted screening tools |
| **Class III** | High | Software that claims to replace professional medical judgment with autonomous diagnoses or treatment recommendations | Autonomous diagnostic AI, treatment-recommending systems |

**For an AI early warning system in an ICU:**
- If the system **assists** clinicians by flagging high-risk patients (decision support) → likely **Class II**
- If the system **autonomously makes** diagnostic or treatment decisions → likely **Class III**
- Most deployed clinical deterioration prediction systems internationally are classified as Class II equivalent

### 1.3 Approval Process Steps

1. **Pre-submission Consultation (2-4 weeks):** Free consultation with TFDA to confirm classification and requirements
2. **Quality System Documentation (QSD) Application (~10 months, can run concurrently):**
   - Establish ISO 13485-compliant Quality Management System (QMS)
   - Abbreviated pathway available for manufacturers with valid audit reports from US FDA, EU Notified Bodies, or Japan PMDA
3. **Product Registration Application:**
   - Submit technical documentation including algorithm description, training data details, validation results
   - Clinical evaluation report (MRMC studies recommended for high-risk products)
   - Cybersecurity documentation
4. **TFDA Review:**
   - Class I: ~1-3 months
   - Class II: ~140 calendar days (~4.5 months), up to 8-12 months with clock-stops
   - Class III / Novel devices: ~220 calendar days (~7 months), up to 12+ months
5. **Registration License Issuance**

### 1.4 Approval Timeline Summary

| Device Class | Typical Timeline | Notes |
|-------------|-----------------|-------|
| Class I (Listing) | ~1 month | Simplified pathway |
| Class I (Standard) | 3-6 months | — |
| Class II | 4.5-12 months | 140 calendar days base, clock-stops for queries |
| Class III / Novel | 7-12+ months | 220 calendar days base, may require Medical Device Committee |
| QSD Process | Up to 10 months | Runs concurrently with product review |
| **Total realistic estimate** | **8-18 months** | **Including QSD + product registration** |

**Fast-track options:**
- **Priority Review:** For products with government research funding and Taiwan-based clinical trials
- **US/EU Predicate Fast-track:** Devices already approved by US FDA or bearing CE marking can leverage existing technical documents and clinical data, significantly reducing review time

### 1.5 August 2025 Updated AI/ML Guidelines — Key Requirements

The most recent TFDA guidelines (effective August 14, 2025) impose several critical requirements:

1. **Local Population Data Requirement:** Test data must represent the target (Taiwanese) population. International manufacturers relying solely on Western training data must supplement with local clinical data.
2. **Clinical Significance of Outputs:** Products must clearly explain the clinical significance and limitations of AI outputs (e.g., risk scores must explain their clinical basis and grading criteria).
3. **Lifecycle Quality Management:** Emphasis on training data management, generalization testing, and quality documentation throughout the product lifecycle — from development through post-market monitoring.
4. **Independent Performance Assessment:** Standalone performance verification required for all AI/ML medical devices.
5. **Clinical Evaluation:** Multi-reader-multi-case (MRMC) study designs recommended. Clinical trial data may be required for high-risk or completely novel products.
6. **Post-Market Change Management:** Core algorithm changes (e.g., swapping the model architecture) trigger a formal approval/variation submission. Minor bug fixes and UI updates are exempt.

### 1.6 Comparison: Taiwan TFDA vs. US FDA vs. EU MDR

| Feature | Taiwan (TFDA) | US (FDA) | EU (MDR) |
|---------|--------------|----------|----------|
| **Regulatory Body** | TFDA under MOHW | FDA | Notified Bodies + European Commission |
| **Classification** | Class I, II, III | Class I, II, III | Class I, IIa, IIb, III (Rule 11) |
| **AI-Specific Guidance** | CADe/CADx guidelines (2025) | PCCP, AI/ML Action Plan | AI Act + MDR requirements |
| **Primary Pathway for AI** | Risk-based registration | 510(k) (85% of AI devices), De Novo | Conformity assessment via Notified Body |
| **Clinical Evidence** | Accepts foreign data; local data increasingly required | Required when risk warrants | CER + PMCF required for all SaMD |
| **Local Data Requirement** | Yes — Taiwanese population data emphasized | No explicit requirement (but FDA wants representative data) | European population representativeness expected |
| **Post-Market Algorithm Updates** | Core changes require approval | PCCP for pre-planned changes | Full conformity reassessment |
| **Regulatory Philosophy** | Risk-based, IMDRF-aligned, pragmatic | Flexible, guidance-driven | Legally binding, prescriptive |
| **Approval Speed** | 4.5-12+ months | 3-12 months (510(k) median ~5 months) | 6-18+ months (Notified Body backlog) |
| **Cross-Recognition** | Accepts US/EU technical docs for fast-track | Standalone system | Standalone system |

**Key insight:** Taiwan's TFDA is generally **more pragmatic** than the EU MDR and **less established** than the US FDA for AI devices, but is rapidly catching up. The 2025 guidelines align Taiwan closely with international standards while maintaining a strong emphasis on local population validation.

### 1.7 Recent AI Medical Devices Approved in Taiwan

As of December 2024:
- **~50 domestically manufactured** AI/ML-based medical devices authorized by TFDA
- **~116 imported** AI/ML-based medical devices authorized
- In 2023 alone, **6 domestically manufactured AI/ML SaMD** obtained market approval
- **76 projects** received consultation from the TFDA Office for AI/ML-based SaMD in 2023

**Notable approved AI devices:**
- **PANCREASaver** (NTUH/NTU): Deep learning CT analysis for pancreatic cancer detection — TFDA approved + US FDA Breakthrough Device Designation
- **EverFortune.AI Cardiothoracic Ratio System** (CMUH): US FDA (K212624) and Taiwan TFDA (007443) approved
- **EverFortune.AI Sepsis Prediction System** (CMUH): Dual US FDA and TFDA licensure
- **ASUS EndoAim**: AI endoscopy system deployed across ~60 Taiwan medical institutions
- **CMUH**: 8 AI models with TFDA approval, 8 more submitted (12 total FDA approvals from Taiwan and US via spin-off EverFortune.AI)

---

## 2. NTUH-Specific AI Deployment

### 2.1 Overview of NTUH

National Taiwan University Hospital (NTUH) is Taiwan's premier teaching hospital, founded in 1895:
- **7,500+ staff**, **2,600+ beds**
- Multiple campuses including specialized branches (children's, cancer, regional)
- Consistently ranked among Asia's top hospitals
- Key participant in Taiwan's national AI healthcare initiatives

### 2.2 AI Systems Deployed or In Development at NTUH

#### A. NTU Medical Genie — AI Clinical Decision Support System
- Integrates **entire electronic medical records** at NTUH: diagnoses, medications, procedures, lab data, imaging, nursing records
- Collects lifestyle and environmental data from 10,000 patients for data enrichment
- Focus areas: genetic diseases, COPD, inherited retinal degenerations
- Combines genomic science with EHR data and AI models

#### B. Large Language Model (LLM) Deployments
NTUH first acquired **NVIDIA AI supercomputers in 2020**, leading to multiple GenAI applications:
- **ICD-10 automatic coding**
- **Automated health check-up report generation**
- **Telemedicine consultation transcription**
- **Emergency voice record reporting**
- **Pathology report key point extraction**
- **Medical record summarization**
- **Unstructured data mining**
- **Medical Q&A**
- All integrated into NTUH's Health Information System (HIS)
- Acquired **two additional supercomputers** in late 2024/2025 for multimodal LLM development

#### C. PANCREASaver — Pancreatic Cancer AI
- Developed with NTU Institute of Applied Mathematical Sciences
- Deep learning analysis of CT scans: auto-delineates pancreas, flags suspicious lesions
- **Directly integrated with NTUH's PACS** (Picture Archiving and Communication System)
- Accessible across departments: gastroenterology, surgery, oncology
- **80% sensitivity** for early-stage tumors (<2 cm), **>90% overall diagnostic accuracy**
- **TFDA approved** + **US FDA Breakthrough Device Designation**

#### D. Federated AI + MRI Platform
- Federated AI-powered generation platform
- Seamless workflow: MRI image processing → diagnostic report generation → clinical decision-making
- Driven by autonomous AI Agents

#### E. Pharmacogenomics Clinical Decision Support
- Next-generation sequencing + gene-drug database
- CDSS covering **22 key pharmacogenes**
- Enables intelligent prescribing and precision medication

#### F. DeepEMR Medical Record Exploration System
- Integrates LLMs with retrieval-augmented generation (RAG)
- Automatically organizes medical records, generates reports, structures research data

#### G. Other AI Tools
- **Oral cancer screening** (smartphone-based, >95% accuracy, <2 minutes)
- **Lung cancer AI platform** (radiomics + clinical data for personalized predictions)
- **AI endoscopy** (ASUS EndoAim deployed at NTUH)

### 2.3 NTUH IT Infrastructure

#### EHR System Architecture
Based on published research (PMID: 22480301), NTUH's healthcare information system features:

- **Multi-tier, Service-Oriented Architecture (SOA):** Integrates heterogeneous legacy systems into a unified enterprise healthcare information system
- **HL7 Messaging Middleware:** HL7 message standard widely adopted for all data exchanges; all services are independent modules enabling flexible deployment
- **DICOM Standard:** For medical imaging data exchange
- **IHE (Integrating the Healthcare Enterprise) Workflow:** Standard clinical workflows
- **Inpatient Information System (IIS):** Core system evaluated for reliability and robustness in high-traffic environments
- **PACS Integration:** PANCREASaver and other imaging AI directly integrated
- **Interoperability:** Open and flexible environment enabling data sharing across NTUH branch hospitals
- **NVIDIA Supercomputers:** For AI model training (acquired 2020, expanded 2024/2025)
- **Private Cloud Platform:** For real-time medical data analysis

#### FHIR Adoption (Emerging)
- Taiwan is transitioning from legacy **HL7 CDA** to **FHIR-based** interoperability
- MOHW hosted the inaugural **International FHIR Server Performance Competition** (December 2025)
- **MedInfo 2025** was hosted in Taipei (August 2025), including a FHIR Master Class
- Taiwan's **My Health Bank** (NHI personal health records) is FHIR-based
- Full FHIR adoption across all institutions remains a work in progress

### 2.4 Published Research on AI at NTUH

Key published areas include:
- AI clinical decision support for precision medicine (NTU Medical Genie project)
- PANCREASaver pancreatic cancer detection (clinical validation trials published)
- LLM applications in healthcare operations
- Pharmacogenomics clinical decision support
- Service-oriented architecture for enterprise healthcare information systems (PMID: 22480301)
- Emergency AI applications (narrative review in PMC: PMC10938302)

---

## 3. Technical Requirements for Clinical Deployment

### 3.1 Real-Time Inference Requirements

| Parameter | Typical Requirement | Notes |
|-----------|-------------------|-------|
| **Inference Latency** | <2 seconds per prediction | A Shanghai hospital found that even 2-second EHR integration latency disrupted workflow; took 8 months to resolve |
| **Prediction Frequency** | Every 15 minutes (Stanford model) to continuous | Stanford's deterioration model updates every 15 minutes |
| **Throughput** | Must handle entire ICU census simultaneously | Typical ICU: 10-50 beds, each generating continuous data streams |
| **Availability** | 99.9%+ uptime | Clinical systems cannot have downtime |
| **Fallback** | Graceful degradation when AI unavailable | System must not crash or block clinical workflow |

**Architecture patterns:**
- **Edge computing:** Lightweight models on bedside monitors for ultra-low latency (arrhythmia detection, etc.)
- **Cloud/on-premise servers:** More complex models (deterioration prediction) running on hospital servers
- **Hybrid:** Edge for real-time vitals processing, server for multi-variable prediction models

### 3.2 EHR Integration Standards

#### Taiwan-Specific Landscape
- **Current standard:** HL7 v2 messaging (widely adopted at NTUH and most Taiwan hospitals)
- **Transitioning to:** HL7 FHIR (government-driven initiative, not yet universal)
- **Medical imaging:** DICOM standard
- **Clinical workflows:** IHE (Integrating the Healthcare Enterprise)

#### Integration Architecture Options
1. **Direct HL7 v2 Interface:** Parse HL7 ADT (admission/discharge/transfer), ORU (observation results), ORM (orders) messages in real-time
2. **FHIR API (emerging):** Standardized REST APIs for patient data access; more "AI-ready" but not yet universally available in Taiwan
3. **Database-level integration:** Direct queries to clinical data warehouse (faster but tighter coupling)
4. **Middleware/Integration Engine:** (e.g., Mirth Connect, Rhapsody) — translates between AI system and hospital HIS

#### Practical Considerations for NTUH Integration
- NTUH uses SOA with HL7 middleware → AI system can subscribe to relevant HL7 message feeds
- PACS integration demonstrated (PANCREASaver) → imaging AI integration is proven
- Real-time vital signs may require separate interface to bedside monitors (e.g., Philips/GE via HL7 or proprietary protocols)

### 3.3 Alert Delivery Mechanisms

| Mechanism | Pros | Cons | Used By |
|-----------|------|------|---------|
| **EHR Pop-up/Banner** | Integrated in clinician workflow; hard to miss | Can contribute to alert fatigue; may interrupt workflow | Epic Sepsis Model, most US hospital AI |
| **Dedicated Dashboard** | Rich visualization; trending; can show contributing factors | Requires clinicians to actively check; may be ignored | Stanford deterioration model, eCART |
| **Mobile Push Notification** | Reaches clinicians anywhere; immediate | Can be lost among other notifications; battery/connectivity | Stanford (mobile alert to care team) |
| **Pager Integration** | Reliable; established workflow in ICUs | Limited information; one-directional | Traditional rapid response systems |
| **Nurse Station Display** | Passive monitoring; color-coded patient list | Requires line-of-sight; not mobile | Some Asian hospital implementations |
| **Combined Approach** | Comprehensive coverage | Complex to implement | Best practice recommendation |

**Best practice (from Stanford and eCART implementations):**
- EHR-integrated alert with contributing factors displayed
- Mobile notification to responsible clinician
- Dashboard for nursing station/charge nurse overview
- Escalation pathway if initial alert not acknowledged within defined time window

### 3.4 Alert Fatigue Management

#### The Problem
- Clinicians receive **50-100+ EHR alerts per day**
- Override rates for medication alerts: **49-96%**
- Epic Sepsis Model: **140,000 alerts in 10 months**, only **13% acknowledged**
- Epic Sepsis Model: only **7% PPV** at recommended threshold

#### What PPV Is Clinically Acceptable?

| PPV Range | Clinical Implication | Examples |
|-----------|---------------------|----------|
| **<10%** | Unacceptable — >90% false alarms, guaranteed alert fatigue | Epic Sepsis Model at some sites |
| **10-20%** | Marginal — high false alarm rate, only viable with strong workflow | Many screening tools in oncology |
| **20-35%** | Acceptable range for ICU early warning systems | Continuous predictive analytics: 24-35% PPV for respiratory-driven alerts |
| **35-50%** | Good — actionable for most clinicians | eCART at Yale (with workflow integration) |
| **>50%** | Excellent — most alerts are true positives | Rare for early warning systems; more typical for diagnostic AI |

**Key insight:** PPV depends critically on **disease prevalence in your specific population**. Even with 95% sensitivity and 95% specificity, at 1% prevalence the PPV is only 16%. Institutions must calculate expected PPV based on their own patient mix.

#### Strategies to Improve PPV and Reduce Alert Fatigue
1. **Threshold Tuning:** Adjust alert thresholds to local population; favor specificity over sensitivity
2. **Tiered Alerting:** Low-risk = dashboard only; medium = nurse notification; high = immediate physician alert
3. **Context-Aware Suppression:** Suppress alerts if clinician has already taken action (e.g., blood cultures ordered)
4. **Workflow Embedding:** Alerts within existing clinical workflow, not separate systems
5. **Acknowledgment + Action Tracking:** Require structured response to alerts; track compliance
6. **Temporal Windowing:** Avoid repeated alerts for same patient within short time window
7. **Human-in-the-Loop Validation:** Charge nurse screens alerts before escalation

### 3.5 Model Monitoring and Drift Detection in Production

#### Types of Drift in Clinical AI

| Drift Type | Description | Clinical Example |
|-----------|-------------|-----------------|
| **Data Drift (Covariate Shift)** | Input data distribution changes | New patient demographics, different monitoring equipment, EMR system upgrade changing data format |
| **Concept Drift** | Relationship between inputs and outcomes changes | New treatment protocols change mortality patterns; COVID-19 pandemic changing ICU case mix |
| **Calibration Drift** | Predicted probabilities no longer match actual outcomes | Model trained on pre-COVID data overestimates risk in post-COVID ICU |
| **Label Drift** | Outcome definitions or coding practices change | ICD code updates, new sepsis definitions (Sepsis-3 vs. earlier) |

#### Monitoring Approaches

1. **Performance Monitoring:**
   - Track AUROC, sensitivity, specificity, PPV, calibration over time
   - Compare against baseline established at deployment
   - **Limitation:** Requires ground truth labels, which may have significant lag in clinical settings

2. **Data Drift Detection:**
   - Monitor input feature distributions (vital signs, lab values, demographics)
   - Statistical tests: KS test, Chi-squared, Population Stability Index (PSI)
   - **Nature Communications (2024) finding:** Performance monitoring alone is NOT a good proxy for data drift detection — direct drift monitoring is essential

3. **Foundation Model Embeddings (MMC+ Framework):**
   - Use pre-trained medical foundation models (e.g., MedImageInsight) to generate high-dimensional embeddings
   - Monitor embedding space concordance over time
   - Detects drift without site-specific training

4. **Calibration Monitoring:**
   - Regularly assess calibration curves
   - Recalibrate when predicted probabilities diverge from observed rates
   - Critical for clinical early warning systems where risk thresholds drive interventions

#### Production Monitoring Architecture

```
Real-time data stream → Feature extraction → Model inference → Alert generation
         ↓                    ↓                    ↓
   Data drift monitor    Feature monitor    Performance tracker
         ↓                    ↓                    ↓
              Monitoring Dashboard / Automated alerts
                            ↓
              Triggered model review / recalibration / retraining
```

#### Key Recommendations
- Monitor both **data drift AND performance** — neither alone is sufficient
- Establish **automated drift detection** with human-in-the-loop escalation
- Plan for **periodic recalibration** (quarterly) even without detected drift
- Maintain a **retraining pipeline** that can be triggered when drift exceeds thresholds
- Document all monitoring results for **regulatory compliance** (TFDA lifecycle QMS requirements)
- Beware of **catastrophic forgetting** when updating models with new data

---

## 4. Ethical and IRB Considerations for Taiwan

### 4.1 IRB Approval Process at NTUH

#### NTUH Research Ethics Committee (REC) Structure
- **Four committees:** REC A, B, C, D (reorganized January 2010, REC D added August 2011)
- **Membership:** Each committee has 19-20 members from diverse backgrounds: chairperson, deputy chairperson, lawyers, physicians, nurses, social workers, laypeople
- **Meeting frequency:** Monthly
- **Quorum:** More than half of members must be present
- **Approval criterion:** More than half of attending members must vote in favor

#### Review Types
Taiwan's national regulations (Regulations for Organization and Operation of Human Research Ethics Review Board) define three review levels:

1. **Exemption from IRB Review:** Minimal-risk research using de-identified data that does not interact with human subjects
2. **Expedited Review:** Minor modifications to previously approved research; research involving minimal risk
3. **Standard (Full Board) Review:** Required for most clinical AI studies involving patient data or clinical outcomes

#### Requirements for AI Studies
For deploying an AI early warning system study at NTUH:
- **Retrospective validation study** (using historical data): May qualify for expedited review or exemption if using de-identified data
- **Prospective clinical trial** (real-time deployment with outcomes assessment): Standard review required
- **Key documentation needed:**
  - Study protocol
  - Informed consent documents
  - Data management plan
  - Risk-benefit analysis
  - Privacy/de-identification methodology
  - Algorithm description and validation results
  - Conflict of interest declarations
- **Post-completion:** Principal investigator must submit final execution and results report
- **Record retention:** 3 years after project completion

#### Transnational Research
If collaborating between international institutions and NTUH:
- If research is conducted exclusively at one institution with a mutual cooperation agreement → one IRB review sufficient
- If conducted at both institutions → separate IRB applications required at each

### 4.2 Taiwan's Personal Data Protection Act (PDPA)

#### Classification of Medical Data
Under the PDPA, the following are classified as **sensitive personal data** with heightened protections:
- Medical records
- Healthcare information
- Genetic data
- Sexual life information
- Physical examination data
- Criminal records

**Collection, processing, and use of sensitive data is generally prohibited** unless specific legal exemptions apply.

#### Legal Bases for Processing Medical Data for AI

| Legal Basis | Description | Applicability to AI Research |
|-------------|-------------|------------------------------|
| **Explicit written consent** | Data subject provides written informed consent | Standard for prospective studies |
| **Required by law** | Processing required by specific legislation | Limited applicability |
| **Government agency statutory duty** | Within necessary scope of statutory duties | Applicable to public hospitals conducting approved research |
| **Non-government entity statutory obligation** | With proper security measures | May apply to academic medical centers |
| **De-identified data for statistical/academic purposes** | Constitutional Court has upheld this basis | Key pathway for retrospective AI research |

#### The Constitutional Court Ruling on Health Data (2022)

**Judgment 111-Hsien-Pan-13** is a landmark ruling concerning the National Health Insurance Research Database:
- **Upheld:** The constitutionality of using de-identified NHIRD data for secondary research purposes
- **Found:** That restrictions on data subject rights for healthcare/public health purposes with de-identified data are proportionate
- **But required:** The legislature to enact stronger safeguards (independent oversight, better de-identification standards)
- **Implication for AI research:** De-identified health data CAN be used for AI research/development, but adequate de-identification is essential

#### De-Identification Requirements

**Current challenges:**
- Taiwan's PDPA lacks clear definitional guidance distinguishing **pseudonymization** from **full anonymization**
- No specific technical standards for health data de-identification (unlike US HIPAA Safe Harbor / Expert Determination methods)
- The Constitutional Court ruling calls for clearer standards, but implementing legislation is pending

**Practical approach for AI research:**
- Remove direct identifiers (name, national ID, address, contact information)
- Apply k-anonymity or differential privacy techniques
- Consider temporal generalization for dates
- Document de-identification methodology for IRB and regulatory review
- Maintain linkage keys separately under strict access controls if re-identification needed for validation

### 4.3 2025 PDPA Amendments

Amendments promulgated November 11, 2025 (not yet in force):
- **Data Breach Notification:** Mandatory notification to data subjects and competent authority
- **Data Protection Officers:** Required for government agencies
- **Personal Data Protection Commission (PDPC):** Independent supervisory authority being established (preparatory office since December 2023; mandated by August 2025 per Constitutional Court ruling)

### 4.4 Taiwan's AI Basic Law (Passed December 23, 2025)

The Artificial Intelligence Basic Act establishes **seven foundational principles:**
1. **Transparency**
2. **Privacy**
3. **Autonomy**
4. **Fairness**
5. **Cybersecurity**
6. **Sustainable Development**
7. **Accountability**

**Current status:** Framework statute — articulates principles and institutional arrangements but detailed implementation regulations are forthcoming. Sector-specific AI governance (including healthcare) will be developed by competent authorities (MOHW/TFDA for healthcare).

### 4.5 Patient Consent for AI-Assisted Care

**Current state in Taiwan:**
- **No specific requirement** yet mandating informed consent specifically for AI-based clinical decision support
- **PDPA consent requirements** apply to the collection and use of patient data (written consent required for sensitive health data)
- **AI Basic Law** establishes principles of transparency and autonomy but lacks implementation details
- **International trend** is toward disclosure of AI use as part of informed consent — Taiwan will likely follow

**Recommended approach for NTUH deployment:**
1. Include AI system use disclosure in general hospital informed consent
2. Provide patient-facing information about the AI early warning system
3. Ensure clinicians understand the AI system's capabilities and limitations (to enable informed clinical communication)
4. Document the AI's role as a decision-support tool (not autonomous decision-maker)
5. Allow patients to request their care not be influenced by AI predictions (opt-out mechanism)

---

## 5. Examples of Successfully Deployed ICU AI Systems Worldwide

### 5.1 Epic Sepsis Model (ESM) — Cautionary Tale

**Deployment:** Used by hundreds of hospitals in the US via Epic EHR

**Reported Problems:**
- **Poor predictive accuracy:** AUC of 0.63 in independent validation (vs. 0.76-0.83 claimed by Epic)
- **Circular predictions:** Model was cueing on whether diagnostic tests/treatments had already been ordered — meaning clinicians already suspected sepsis before the AI flagged it
- **Only 53%** of sepsis patients received higher risk scores when predictions were restricted to before blood cultures were ordered
- **Massive alert volume:** 140,000 alerts in 10 months; only 13% acknowledged
- **7% PPV** at recommended alert threshold — meaning 93% of alerts were false alarms
- **ICU staff stress:** Constant alert stream increased stress; counterproductive in post-operative care
- **Fundamental flaw:** The "Time Zero" problem — sepsis onset is unknown, so models are trained on proxies that rely on clinical recognition, leading to circular predictions

**Lessons for Taiwan deployment:**
- Independent, external validation on local population data is essential (aligns with TFDA local data requirements)
- PPV must be validated at the local hospital's disease prevalence
- Avoid proprietary "black box" models with non-peer-reviewed performance claims
- Alert workflow design is as important as model performance

### 5.2 eCART — University of Chicago

**Approach:** General clinical deterioration detection (not sepsis-specific)

**Key design principles:**
- Predicts **discrete deterioration events** (ICU transfer, death) rather than sepsis specifically
- Avoids the "Time Zero" definitional problem of sepsis models
- ~40% of detected deterioration events are due to sepsis, but the model is not limited to sepsis
- Trained on objective, time-stamped outcomes

**Deployment results at Yale New Haven Health:**
- First system to achieve **universal adoption** of a clinical deterioration tool in 10+ years
- Nurses complete sepsis screen and/or worry assessment in **>90%** of patients with eCART elevations
- Yale selected eCART after comparing 6 general early warning scores in a cohort of **>360,000 patients**

**Key lesson:** General deterioration prediction may outperform disease-specific prediction models in clinical utility.

### 5.3 Stanford Health Care — AI Clinical Deterioration System

**Architecture:**
- ML model updates predictions on hospitalized patients **every 15 minutes**
- Validated on Stanford's own institutional data
- Signals integrated into EHR with **full transparency** (contributing factors displayed)
- **Mobile alert** to care team
- Acts as an "objective assessor of risk" facilitating care coordination

**Stanford-Harvard State of Clinical AI Report (2026) findings:**
- Wearable-based AI can predict deterioration **8-24 hours** before standard hospital alerts
- Strongest AI results appear in **prediction tasks** (risk scores, early warning)
- **Caution:** Clinicians sometimes follow incorrect AI recommendations even when errors are detectable — suggesting over-reliance risk

### 5.4 Asian Hospital AI Deployments

#### Taiwan — China Medical University Hospital (CMUH)
**"AI Hospital of the Future"** — the most extensively AI-deployed hospital in Taiwan:
- **Hundreds of AI algorithms** deployed across **12 hospitals** on Microsoft Azure
- **8 TFDA-approved** AI models, 8 more submitted (12 total US FDA + TFDA approvals via EverFortune.AI)
- **Specific deployed systems:**
  - **i.A.M.S (Intelligent Antimicrobial System):** Personalized antibiograms, sepsis/mortality prediction, MDR organism detection, antibiotic CDSS → **25% mortality reduction, 30% antibiotic cost reduction, 50% antibiotic use reduction**
  - **ARDiTeX (ARDS Diagnostic Tool):** AI-assisted ARDS diagnosis → **20% survival rate improvement**
  - **ECG/STEMI Detection in Ambulances:** AI detects heart attacks en route → **halved door-to-treatment time**
  - **Drug-resistant bacteria detection:** Identifies 6 types of MDR bacteria in 1 hour
  - **10 AI outpatient clinics** since 2019 (cardiology, nephrology, pulmonology, breast surgery, pediatrics, ophthalmology, precision medicine, health screening)
- **Awards:** HIMSS Davies Award (only hospital in Asia, 2023); Newsweek World's Best Smart Hospitals (2024); HIMSS DHI #3 globally (2022)

#### Taiwan — Taipei Veterans General Hospital (VGHTPE)
- **TvHEWS (Time-varying Hemodynamic Early Warning Score):** AI model predicting hemodynamic instability in ICU patients
- Uses 24 time-varying ML models for continuous risk assessment over 24-hour periods
- Trained on VGHTPE 2010-2021 data, validated prospectively on 2022 data and externally on MIMIC IV
- Addresses early detection of hemodynamic instability through continuous, dynamic risk assessment

#### Singapore
- **BRAIN Platform:** Centralized AI platform deployed across **all large Singapore public hospitals since April 2017** (first-of-its-kind nationwide deployment)
  - Triggers prediction models daily for all inpatients
  - Generates patient risk scores
- **SERA Algorithm:** Sepsis prediction using structured + unstructured clinical notes
  - AUC 0.94, sensitivity 0.87, specificity 0.87 at 12 hours before onset
  - Potential to increase early detection by 32% and reduce false positives by 17%
- **Changi General Hospital:** Developing AI algorithms for patient deterioration prediction as part of smart remote monitoring
- **Ng Teng Fong General Hospital:** AI-predicted bed demand for flu outbreaks
- Singapore's approach emphasizes **robust government-hospital-startup-research collaboration**

#### South Korea
- **National AI Health Infrastructure:** Government funding hospital-based AI verification programs (20+ projects)
- **Emergency Care AI:** $15M+ investment in AI-based patient classification and transfer systems
- **Regulatory Leadership:** First to release Guidelines for Generative AI Medical Devices (2025)
- Major AI companies: **Lunit, VUNO, Coreline Soft** — expanding globally in medical imaging AI
- **Korea Advanced Research Projects Agency for Health (KRAPH):** Driving AI emergency care initiatives

#### Japan
- Aging population driving strong demand for healthcare AI
- Active regulatory frameworks (PMDA)
- Less publicly documented ICU AI early warning system deployments compared to Korea/Singapore/Taiwan
- Growing investment in AI for elderly care and hospital efficiency

### 5.5 Meta-Analysis Evidence (2025)

A systematic review and meta-analysis in *BMC Medical Informatics and Decision Making* found:
- **5 prospectively validated studies** showed AI early warning systems **significantly reduce in-hospital and 30-day mortality**
- In one study of **13,000+ admissions**, non-palliative in-hospital deaths fell from **2.1% to 1.6%** after AI EWS deployment
- **ICU length of stay increased** (suggesting patients were being identified and treated earlier)
- **Hospital stays shortened** overall
- **Clinician adherence** to AI warnings remains a challenge

### 5.6 FDA-Authorized Sepsis AI Tool (2024)

**Sepsis ImmunoScore** — First FDA-authorized AI diagnostic tool for sepsis (De Novo, April 2024):
- Uses host immune response biomarkers (not just EHR data)
- AUC: 0.85 (derivation), 0.80 (internal validation), 0.81 (external validation)
- Risk categories predict in-hospital mortality: low (0.0%), medium (1.9%), high (8.7%), very high (18.2%)
- Represents a shift toward biomarker-based rather than EHR-pattern-based sepsis prediction

---

## Summary: Key Recommendations for NTUH ICU AI Deployment

### Regulatory Pathway
1. **Classify the system** — likely Class II if decision-support, Class III if autonomous
2. **Plan for 8-18 months** total approval time (QSD + product registration)
3. **Collect Taiwanese population validation data** — mandatory under 2025 guidelines
4. **Consider Priority Review** if government-funded with Taiwan-based clinical trials
5. **Leverage US/EU predicate** if available for fast-track

### Technical Integration
1. **Interface via HL7 v2 middleware** (NTUH's current infrastructure) with FHIR readiness
2. **Target <2 second inference latency** with 15-minute prediction refresh cycle
3. **Integrate alerts into existing HIS workflow** — not a standalone system
4. **Implement tiered alerting** to manage alert fatigue
5. **Deploy model monitoring** from day one — data drift + performance tracking

### Ethical/Regulatory Compliance
1. **Submit to NTUH REC** for standard review (prospective study) or expedited review (retrospective)
2. **Comply with PDPA** — written consent for sensitive data; document de-identification
3. **Plan for AI Basic Law** implementation regulations
4. **Disclose AI use** to patients as part of informed consent process

### Learning from Others
1. **Avoid the Epic Sepsis Model's mistakes** — validate independently, monitor PPV at local prevalence, design workflows thoughtfully
2. **Emulate CMUH's approach** — start with specific clinical domains, integrate tightly with existing workflows, pursue TFDA approval
3. **Follow Singapore's model** — centralized AI platform, nationwide standardization, government backing
4. **Adopt Stanford's transparency** — show contributing factors, mobile alerts, 15-minute refresh

---

## Sources

### Taiwan FDA / Regulatory
- [Taiwan Regulates Software as Medical Device — Pacific Bridge Medical](https://www.pacificbridgemedical.com/uncategorized/regulations-of-software-as-a-medical-device-in-taiwan/)
- [TFDA New AI Medical Device Guidelines — Cisema](https://cisema.com/en/taiwan-fda-updates-ai-medical-device-technical-guidelines/)
- [Digital Health Laws and Regulations 2025 Taiwan — ICLG](https://iclg.com/practice-areas/digital-health-laws-and-regulations/taiwan)
- [Healthcare AI 2025 Taiwan — Chambers](https://practiceguides.chambers.com/practice-guides/healthcare-ai-2025/taiwan/trends-and-developments)
- [Taiwan Medical Device Registration — Emergo by UL](https://www.emergobyul.com/resources/taiwanese-tfda-regulatory-approval-process-medical-and-ivd-devices)
- [MOHW Advancing Regulatory System of AI Medical Devices](https://www.mohw.gov.tw/cp-4745-64039-2.html)
- [TFDA 2023 Annual Report](https://www.fda.gov.tw/upload/ebook/AnnuaReport/2023/2023_E/files/basic-html/page76.html)
- [Medical Device Registration in Taiwan — Pacific Bridge Medical](https://www.pacificbridgemedical.com/regulatory-services/medical-device/product-registration/taiwan/)
- [Taiwan Medical Device Registration — Asia Actual](https://asiaactual.com/taiwan/medical-device-registration/)
- [Guide to Medical Device Approval Process in Taiwan — Credevo](https://credevo.com/articles/2024/01/15/guide-to-medical-device-approval-process-in-taiwan-regulations-requirements-steps/)
- [Taiwan TFDA Revises AI/ML CADe/CADx Guidance — NordPacific](https://nordpacificmed.com/taiwan-tfda-revises-guidance-for-ai-ml-based-cade-and-cadx-medical-device-registration/)
- [Advancements in Clinical Evaluation and Regulatory Frameworks for AI-Driven SaMD — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11655112/)
- [Regulatory Frameworks for AI-Enabled Medical Device Software in China — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11319888/)
- [A Decade of Review in Global AI Medical Device Regulation — Frontiers](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1630408/full)

### NTUH & Taiwan Hospital AI
- [NTU Medical Genie — AI Decision Support System](http://mahc.ntu.edu.tw/en/research_view.php?id=1)
- [NTUH Multimodal LLM Development — Healthcare IT News](https://www.healthcareitnews.com/news/asia/national-taiwan-university-hospital-go-multimodal-large-ai-development)
- [NTUH Pancreatic Cancer Imaging AI — Healthcare IT News](https://www.healthcareitnews.com/news/asia/national-taiwan-university-hospital-fully-deploys-pancreatic-cancer-imaging-ai)
- [Inside Taiwan's AI Hospital of the Future (CMUH) — Microsoft](https://news.microsoft.com/source/asia/features/inside-taiwans-ai-hospital-of-the-future/)
- [CMUH Smart Hospital — CMUH](https://cmuh.cmu.edu.tw/CMUHPagesDetail/SHDS/SmartHospital_new)
- [CMUH AI Sepsis Prediction — Healthcare+ Expo](https://expo.taiwan-healthcare.org/en/news_detail.php?REFDOCID=0r1mpaby9wmpb224)
- [CMUH Gen AI Upgrade — PR Newswire](https://www.prnewswire.com/news-releases/china-medical-university-hospital-cmuh-in-taiwan-upgrades-smart-healthcare-with-gen-ai-302062857.html)
- [CMUH HIMSS Validations — Healthcare IT News](https://www.healthcareitnews.com/news/asia/china-medical-university-hospital-achieves-three-himss-validations)
- [ASUS Healthcare Hub — ASUS Pressroom](https://press.asus.com/news/press-releases/asus-healthcare-orchestration-hub-taiwan-expo/)
- [Taiwan MOHW AI Centers — Healthcare+ B2B](https://www.taiwan-healthcare.org/en/news-detail?id=0sl0jht9q78wwps2)
- [AI in Taiwan Emergency Medicine — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10938302/)
- [NTUH SOA Healthcare Information System — PubMed](https://pubmed.ncbi.nlm.nih.gov/22480301/)

### FHIR / Interoperability in Taiwan
- [Taiwan FHIR-based PHR — MDPI](https://www.mdpi.com/2071-1050/13/1/198)
- [Phison Wins MOHW FHIR Server Competition — BusinessWire](https://www.businesswire.com/news/home/20251224728804/en/Phison-Wins-Two-Awards-at-Taiwan-MOHWs-First-International-FHIR-Server-Performance-Competition)
- [MedInfo 2025 in Taipei](https://medinfo2025.org/)
- [State of FHIR 2025 — Fire.ly](https://fire.ly/blog/the-state-of-fhir-in-2025/)

### Data Protection & Ethics
- [Data Protection Laws 2025-2026 Taiwan — ICLG](https://iclg.com/practice-areas/data-protection-laws-and-regulations/taiwan)
- [Taiwan PDPA Major Amendments — Jones Day](https://www.jonesday.com/en/insights/2025/12/taiwan-passes-major-amendments-to-the-personal-data-protection-act)
- [Taiwan PDPA Amendment — Baker McKenzie](https://insightplus.bakermckenzie.com/bm/data-technology/taiwan-amendment-to-personal-data-protection-act)
- [Secondary Use of Health Data Taiwan & EU — PMC](https://ncbi.nlm.nih.gov/pmc/articles/PMC11250748)
- [Data Protection & Privacy 2025 Taiwan — Chambers](https://practiceguides.chambers.com/practice-guides/data-protection-privacy-2025/taiwan)
- [Taiwan PDPA — Securiti](https://securiti.ai/solutions/taiwan-personal-data-protection-act/)
- [Taiwan AI Basic Law — White & Case](https://www.whitecase.com/insight-our-thinking/ai-watch-global-regulatory-tracker-taiwan)
- [Taiwan AI Governance — STLI](https://stli.iii.org.tw/en/article-detail.aspx?no=105&tp=2&i=168&d=9199)
- [Taiwan AI Basic Law — Lexology](https://www.lexology.com/library/detail.aspx?g=f63ec85f-4034-4e94-aabe-ffdd994c3f0f)
- [AI Laws Taiwan — Global Legal Insights](https://www.globallegalinsights.com/practice-areas/ai-machine-learning-and-big-data-laws-and-regulations/taiwan/)
- [NTUH REC — NTUH](https://www.ntuh.gov.tw/RECO/Fpage.action?muid=11&fid=1952)
- [NTU Research Ethics Review Q&A](https://ord.ntu.edu.tw/w/ordntuEN/info_21011422021187741)
- [Taiwan Human Research Ethics Regulations](https://law.moj.gov.tw/Eng/LawClass/LawAll.aspx?PCode=L0020179)

### Deployed ICU AI Systems
- [Epic Sepsis Model Validation — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10317482/)
- [Epic Sepsis Model Cribbing Clinician Suspicions — UMich](https://news.umich.edu/widely-used-ai-tool-for-early-sepsis-detection-may-be-cribbing-doctors-suspicions/)
- [Epic Sepsis Model Lacking Predictive Power — Healthcare IT News](https://www.healthcareitnews.com/news/research-suggests-epic-sepsis-model-lacking-predictive-power)
- [End User Experience of Epic Sepsis System — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11458550/)
- [Epic Algorithm Overhaul — STAT News](https://www.statnews.com/2022/10/24/epic-overhaul-of-a-flawed-algorithm/)
- [eCART / Sepsis Alliance Webinar Recap — AgileMD](https://www.agilemd.com/post/sepsis-alliance-webinar-recap-sepsis-prediction-models-yales-ecart-case-study)
- [Stanford AI for Clinical Deterioration — Healthcare IT News](https://www.healthcareitnews.com/news/stanford-health-uses-ai-reduce-clinical-deterioration-events)
- [Stanford-Harvard State of Clinical AI Report — Stanford Medicine](https://med.stanford.edu/medicine/news/current-news/standard-news/clinical-ai-has-boomed.html)
- [AI-Powered EWS Meta-Analysis — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12131336/)
- [Sepsis ImmunoScore FDA Authorization — NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2400867)
- [TvHEWS Dynamic Early Warning — Springer](https://link.springer.com/article/10.1186/s13054-025-05553-x)

### Model Monitoring / Drift
- [Empirical Data Drift Detection in Medical Imaging — Nature Communications](https://www.nature.com/articles/s41467-024-46142-w)
- [MMC+ Scalable Drift Monitoring — arXiv](https://arxiv.org/abs/2410.13174)
- [Healthcare Concept Drift Survey — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10150933/)
- [Sepsis Patient Safety Practices — NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK603406/)

### Asian Hospital AI
- [Singapore AI in Public Health — Healthcare IT News](https://www.healthcareitnews.com/news/asia/behind-singapores-widespread-ai-adoption-public-health)
- [Singapore Building Trust in AI — WEF](https://www.weforum.org/stories/2025/09/singapore-healthcare-ai/)
- [Changi General Hospital AI — Healthcare IT News](https://www.healthcareitnews.com/news/asia/changi-general-hospital-developing-ai-algorithms-predict-patient-deterioration)
- [Korea AI Health Data Infrastructure — Healthcare IT News](https://www.healthcareitnews.com/news/asia/korea-building-national-ai-ready-health-data-infrastructure)
- [Korea Emergency AI Platform — Healthcare IT News](https://www.healthcareitnews.com/news/asia/korea-pilots-ambulance-emergency-platform-built-10-ai-models)
- [AI Governance in Hospitals — Korea Herald](https://www.koreaherald.com/article/10636350)
- [AI-Powered Healthcare Asia Pacific — IDC](https://blogs.idc.com/2025/07/11/ai-powered-healthcare-in-asia-pacific-whats-next-for-2025-and-beyond/)
