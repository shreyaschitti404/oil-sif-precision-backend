"""
SIH 26165 — SIF Precursor Detection Engine
Improved single-file prototype, preserving the original architecture.

Purpose:
    Detect SIF precursors in safety reports, extract relevant safety factors,
    and transparently prioritize events for human review.

This is a decision-support prototype. Model outputs are not definitive safety
conclusions and do not replace HSE/domain review.

IMPORTANT:
    - The included model artifacts may be trained on synthetic prototype data; such metrics are not evidence of real-world OIL performance.
    - Training requires a genuinely labelled dataset with:
          report_text, sif_label
    - Rule-based activity/LSR/barrier/exposure extraction is separate from the
      learned Transformer model.
    - Priority weights are configurable prototype weights and require HSE/domain
      validation. They are not scientifically validated OIL risk weights.

Supported CLI:
    python sih26165_sif_engine_full.py --mode train --data train.csv
    python sih26165_sif_engine_full.py --mode predict --text "..."
    python sih26165_sif_engine_full.py --mode predict-csv --data new_reports.csv
    python sih26165_sif_engine_full.py --mode api

Recommended installation:
    pip install -r requirements.txt

Optional model:
    --model microsoft/deberta-v3-base
Default:
    distilbert-base-uncased
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)

# ============================================================
# LOGGING / CONFIG
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("sih26165")

DEFAULT_MODEL = "distilbert-base-uncased"
SUPPORTED_MODELS = {
    "distilbert-base-uncased",
    "microsoft/deberta-v3-base",
}

DEFAULT_OUTPUT_DIR = "./sif_model"
DEFAULT_MAX_LENGTH = 256
DEFAULT_SEED = 42

OVERFIT_F1_GAP = 0.10
OVERFIT_LOSS_GAP = 0.20
UNDERFIT_F1 = 0.70

MAX_API_TEXT_CHARS = 20_000
MAX_API_BATCH = 500

# These are transparent prototype weights only.
DEFAULT_PRIORITY_WEIGHTS = {
    "sif": 0.40,
    "activity": 0.20,
    "barrier": 0.30,
    "exposure": 0.10,
}

# ============================================================
# DOMAIN TAXONOMIES — PRESERVED FROM ORIGINAL PROJECT
# ============================================================

LIFE_SAVING_RULES = [
    "Bypassing Safety Controls",
    "Confined Space",
    "Driving",
    "Energy Isolation",
    "Hot Work",
    "Line of Fire",
    "Safe Mechanical Lifting",
    "Work Authorisation",
    "Working at Height",
]

ACTIVITIES = [
    "Confined Space Entry",
    "Driving",
    "Vehicle Movement",
    "Energy Isolation / LOTO",
    "Electrical Work",
    "Hot Work",
    "Welding",
    "Gas Cutting",
    "Mechanical Maintenance",
    "Rotating Equipment Maintenance",
    "Pressure Testing",
    "Line Breaking",
    "Pipeline Work",
    "Drilling",
    "Well Intervention",
    "Well Servicing",
    "Workover",
    "Rig Operations",
    "Crane / Lifting",
    "Rigging",
    "Material Handling",
    "Excavation",
    "Trenching",
    "Scaffolding",
    "Working at Height",
    "Roof Work",
    "Tank Cleaning",
    "Chemical Handling",
    "Process Operations",
    "Plant Maintenance",
    "Construction",
    "Demolition",
    "Road Work",
    "Manual Handling",
    "Laboratory Work",
    "Inspection",
    "Testing",
    "Commissioning",
    "Shutdown / Turnaround",
    "Other",
]

BARRIER_FAILURES = [
    "Safety interlock bypassed",
    "Alarm override",
    "Guard / protective device removed",
    "Safety trip defeated",
    "Gas testing absent",
    "Entry permit missing",
    "Continuous gas monitoring absent",
    "Rescue arrangement not confirmed",
    "Unauthorized entry",
    "Ventilation inadequate",
    "Seatbelt not used",
    "Speed limit exceeded",
    "Reversing without spotter",
    "Journey plan not followed",
    "Mobile phone use while driving",
    "Vehicle-pedestrian segregation inadequate",
    "LOTO not applied",
    "Zero energy not verified",
    "Wrong equipment isolated",
    "Residual energy not released",
    "Isolation point not identified",
    "Test-before-touch not performed",
    "Hot work permit missing",
    "Flammables not removed",
    "Gas test not completed",
    "Fire watch absent",
    "Ignition source not controlled",
    "Combustible area not screened",
    "Person inside exclusion zone",
    "Stored energy not controlled",
    "Snap-back zone not established",
    "Body near moving equipment",
    "Pressurized line opened unsafely",
    "Dropped-object zone not controlled",
    "Lifting plan absent",
    "Load not secured",
    "Lifting equipment inspection overdue",
    "Exclusion zone not established",
    "Load capacity exceeded",
    "Unsuitable rigging",
    "Unauthorized person in lifting zone",
    "Permit missing",
    "Permit conditions not followed",
    "Permit expired",
    "Scope changed without reauthorization",
    "Work started before permit approval",
    "Issuer/performer verification missing",
    "Fall protection not connected",
    "Guardrail missing",
    "Unsafe access",
    "Anchor point not verified",
    "Open edge not protected",
    "Tools/materials not secured against falling",
    "No significant barrier failure identified",
]

# Each entry contains positive phrases and phrases that indicate the control
# was actually present. The positive/negative context check prevents obvious
# false positives such as "gas testing was completed".
BARRIER_PATTERNS = {
    "Safety interlock bypassed": {
        "positive": ["interlock bypassed", "interlock bypass", "interlock defeated",
                     "interlock overridden", "interlock was bypassed"],
        "negative": ["interlock was not bypassed", "interlock not bypassed",
                     "interlock remained active"],
    },
    "Alarm override": {
        "positive": ["alarm override", "alarm bypass", "alarm disabled",
                     "alarm was overridden"],
        "negative": ["alarm not overridden", "alarm remained enabled"],
    },
    "Guard / protective device removed": {
        "positive": ["guard removed", "guard was removed", "protective device removed",
                     "guard missing"],
        "negative": ["guard was installed", "guard was in place", "guard not removed"],
    },
    "Gas testing absent": {
        "positive": ["no gas test", "gas test not done", "without gas testing",
                     "without gas test", "gas testing absent", "did not conduct gas test",
                     "gas test was not conducted", "gas test not completed"],
        "negative": ["gas testing completed", "gas test completed", "gas testing was done",
                     "gas test was done", "gas test was completed"],
    },
    "Entry permit missing": {
        "positive": ["entry permit missing", "no entry permit", "without entry permit"],
        "negative": ["entry permit obtained", "entry permit available"],
    },
    "Continuous gas monitoring absent": {
        "positive": ["continuous gas monitoring absent", "continuous gas monitoring was absent",
                     "gas monitor not available", "continuous monitoring not provided"],
        "negative": ["continuous gas monitoring available", "continuous gas monitoring provided"],
    },
    "Rescue arrangement not confirmed": {
        "positive": ["rescue arrangement not confirmed", "rescue plan not confirmed",
                     "rescue arrangement absent"],
        "negative": ["rescue arrangement confirmed", "rescue plan confirmed"],
    },
    "Unauthorized entry": {
        "positive": ["unauthorized entry", "entered without authorization", "unauthorised entry"],
        "negative": ["authorized entry", "authorised entry"],
    },
    "Ventilation inadequate": {
        "positive": ["ventilation inadequate", "inadequate ventilation", "poor ventilation"],
        "negative": ["ventilation adequate", "adequate ventilation"],
    },
    "Seatbelt not used": {
        "positive": ["seatbelt not used", "seat belt not worn", "without seatbelt",
                     "without seat belt", "seatbelt was not used"],
        "negative": ["seatbelt used", "seat belt worn", "seatbelt was used"],
    },
    "Speed limit exceeded": {
        "positive": ["overspeed", "speed limit exceeded", "speeding", "exceeded the speed limit"],
        "negative": ["within speed limit", "speed limit was followed"],
    },
    "Reversing without spotter": {
        "positive": ["reversing without spotter", "reverse without spotter",
                     "reversing without a spotter", "no spotter while reversing"],
        "negative": ["spotter present while reversing", "reversing with spotter"],
    },
    "Journey plan not followed": {
        "positive": ["journey plan not followed", "journey plan was not followed",
                     "deviated from journey plan"],
        "negative": ["journey plan followed", "journey plan was followed"],
    },
    "Mobile phone use while driving": {
        "positive": ["mobile phone while driving", "phone while driving",
                     "using phone while driving"],
        "negative": ["phone not used while driving", "mobile phone was not used"],
    },
    "Vehicle-pedestrian segregation inadequate": {
        "positive": ["vehicle pedestrian segregation inadequate",
                     "poor vehicle pedestrian segregation",
                     "vehicle-pedestrian segregation inadequate"],
        "negative": ["vehicle pedestrian segregation adequate",
                     "vehicle-pedestrian segregation adequate"],
    },
    "LOTO not applied": {
        "positive": ["loto not applied", "lockout tagout not applied",
                     "without loto", "without lockout tagout",
                     "lockout/tagout not applied"],
        "negative": ["loto applied", "loto was applied", "lockout tagout applied",
                     "lockout/tagout applied"],
    },
    "Zero energy not verified": {
        "positive": ["zero energy not verified", "zero energy was not verified",
                     "energy not verified", "zero energy verification not done"],
        "negative": ["zero energy verified", "zero energy was verified"],
    },
    "Wrong equipment isolated": {
        "positive": ["wrong equipment isolated", "wrong line isolated",
                     "incorrect equipment isolated"],
        "negative": ["correct equipment isolated", "correct line isolated"],
    },
    "Residual energy not released": {
        "positive": ["residual energy not released", "residual energy remained",
                     "stored pressure not released", "residual pressure remained"],
        "negative": ["residual energy released", "residual energy was released",
                     "stored pressure released"],
    },
    "Isolation point not identified": {
        "positive": ["isolation point not identified", "isolation point unidentified"],
        "negative": ["isolation point identified", "isolation point was identified"],
    },
    "Test-before-touch not performed": {
        "positive": ["test before touch not performed", "test-before-touch not performed",
                     "test before touch was not performed"],
        "negative": ["test before touch performed", "test-before-touch performed"],
    },
    "Hot work permit missing": {
        "positive": ["hot work permit missing", "no hot work permit",
                     "without hot work permit"],
        "negative": ["hot work permit obtained", "hot work permit available"],
    },
    "Flammables not removed": {
        "positive": ["flammables not removed", "flammable materials not removed",
                     "flammables remained"],
        "negative": ["flammables removed", "flammable materials removed"],
    },
    "Gas test not completed": {
        "positive": ["gas test not completed", "gas test incomplete",
                     "gas testing incomplete"],
        "negative": ["gas test completed", "gas testing completed"],
    },
    "Fire watch absent": {
        "positive": ["fire watch absent", "no fire watch", "without fire watch"],
        "negative": ["fire watch present", "fire watch assigned"],
    },
    "Ignition source not controlled": {
        "positive": ["ignition source not controlled", "ignition source uncontrolled"],
        "negative": ["ignition source controlled", "ignition source was controlled"],
    },
    "Combustible area not screened": {
        "positive": ["combustible area not screened", "combustible area was not screened"],
        "negative": ["combustible area screened", "combustible area was screened"],
    },
    "Person inside exclusion zone": {
        "positive": ["person inside exclusion zone", "worker inside exclusion zone",
                     "person in exclusion zone", "worker in exclusion zone"],
        "negative": ["no person inside exclusion zone", "person remained outside exclusion zone"],
    },
    "Stored energy not controlled": {
        "positive": ["stored energy not controlled", "stored energy uncontrolled"],
        "negative": ["stored energy controlled", "stored energy was controlled"],
    },
    "Snap-back zone not established": {
        "positive": ["snap-back zone not established", "snap back zone not established",
                     "snap-back zone not marked"],
        "negative": ["snap-back zone established", "snap back zone established"],
    },
    "Body near moving equipment": {
        "positive": ["body near moving equipment", "hand near moving equipment",
                     "person near moving equipment", "body part near moving"],
        "negative": ["body kept clear", "person kept clear of moving equipment"],
    },
    "Pressurized line opened unsafely": {
        "positive": ["pressurized line opened", "opened pressurized line unsafely",
                     "pressure line opened unsafely"],
        "negative": ["line depressurized before opening", "line was depressurized"],
    },
    "Dropped-object zone not controlled": {
        "positive": ["dropped object zone not controlled", "dropped-object zone not controlled",
                     "dropped object zone not established"],
        "negative": ["dropped object zone controlled", "dropped-object zone controlled"],
    },
    "Lifting plan absent": {
        "positive": ["lifting plan absent", "no lifting plan", "without lifting plan"],
        "negative": ["lifting plan available", "lifting plan approved"],
    },
    "Load not secured": {
        "positive": ["load not secured", "load unsecured", "load was not secured"],
        "negative": ["load secured", "load was secured"],
    },
    "Lifting equipment inspection overdue": {
        "positive": ["lifting equipment inspection overdue", "lifting inspection overdue",
                     "lifting equipment inspection expired"],
        "negative": ["lifting equipment inspection current", "lifting equipment inspected"],
    },
    "Exclusion zone not established": {
        "positive": ["exclusion zone not established", "no exclusion zone",
                     "exclusion zone not set"],
        "negative": ["exclusion zone established", "exclusion zone was established"],
    },
    "Load capacity exceeded": {
        "positive": ["load capacity exceeded", "capacity exceeded", "overloaded lift"],
        "negative": ["within load capacity", "load capacity not exceeded"],
    },
    "Unsuitable rigging": {
        "positive": ["unsuitable rigging", "incorrect rigging", "improper rigging"],
        "negative": ["suitable rigging", "correct rigging"],
    },
    "Unauthorized person in lifting zone": {
        "positive": ["unauthorized person in lifting zone", "person in lifting zone without authorization"],
        "negative": ["authorized person in lifting zone"],
    },
    "Permit missing": {
        "positive": ["permit missing", "without permit", "no permit"],
        "negative": ["permit available", "permit obtained", "permit approved"],
    },
    "Permit conditions not followed": {
        "positive": ["permit conditions not followed", "permit condition violated",
                     "permit conditions violated"],
        "negative": ["permit conditions followed", "permit conditions were followed"],
    },
    "Permit expired": {
        "positive": ["permit expired", "expired permit", "permit had expired"],
        "negative": ["permit valid", "permit was valid"],
    },
    "Scope changed without reauthorization": {
        "positive": ["scope changed without reauthorization", "scope changed without authorization",
                     "changed scope without reauthorization"],
        "negative": ["scope change authorized", "scope change was authorized"],
    },
    "Work started before permit approval": {
        "positive": ["work started before permit approval", "started work before permit approval",
                     "work began before permit approval"],
        "negative": ["work started after permit approval", "permit approved before work"],
    },
    "Issuer/performer verification missing": {
        "positive": ["issuer performer verification missing", "issuer/performer verification missing",
                     "permit verification missing"],
        "negative": ["issuer performer verification completed", "issuer/performer verification completed"],
    },
    "Fall protection not connected": {
        "positive": ["fall protection not connected", "harness not connected",
                     "fall arrest not connected"],
        "negative": ["fall protection connected", "harness connected"],
    },
    "Guardrail missing": {
        "positive": ["guardrail missing", "no guardrail", "guard rail missing"],
        "negative": ["guardrail installed", "guardrail in place"],
    },
    "Unsafe access": {
        "positive": ["unsafe access", "unsafe ladder", "unsafe platform"],
        "negative": ["safe access", "safe ladder", "safe platform"],
    },
    "Anchor point not verified": {
        "positive": ["anchor point not verified", "anchor point was not verified"],
        "negative": ["anchor point verified", "anchor point was verified"],
    },
    "Open edge not protected": {
        "positive": ["open edge", "unprotected edge", "open edge not protected"],
        "negative": ["edge protected", "open edge protected"],
    },
    "Tools/materials not secured against falling": {
        "positive": ["tools falling", "materials falling", "objects could fall",
                     "tools not secured against falling", "materials not secured against falling"],
        "negative": ["tools secured", "materials secured against falling"],
    },
}

# Activity phrases. Multiple matches are returned.
ACTIVITY_PATTERNS = {
    "Confined Space Entry": ["confined space", "tank entry", "vessel entry", "manhole entry"],
    "Working at Height": ["working at height", "height work", "fall protection", "elevated work"],
    "Crane / Lifting": ["crane", "lifting", "lifted load", "lifting operation"],
    "Rigging": ["rigging", "rigger", "lifting gear"],
    "Driving": ["driving", "driver", "while driving"],
    "Vehicle Movement": ["vehicle movement", "vehicle reversing", "reversing", "truck movement"],
    "Energy Isolation / LOTO": ["loto", "lockout", "tagout", "energy isolation", "equipment isolation"],
    "Electrical Work": ["electrical work", "electrical maintenance", "live electrical", "electrical panel"],
    "Hot Work": ["hot work", "hot-work"],
    "Welding": ["welding", "welder"],
    "Gas Cutting": ["gas cutting", "oxy cutting", "flame cutting"],
    "Mechanical Maintenance": ["mechanical maintenance", "mechanical repair"],
    "Rotating Equipment Maintenance": ["rotating equipment", "pump maintenance", "compressor maintenance"],
    "Pressure Testing": ["pressure testing", "hydrotest", "hydro test", "pressure test"],
    "Line Breaking": ["line breaking", "line break", "opened line", "breaking containment"],
    "Pipeline Work": ["pipeline", "pipeline work"],
    "Drilling": ["drilling", "drill rig"],
    "Well Intervention": ["well intervention"],
    "Well Servicing": ["well servicing"],
    "Workover": ["workover"],
    "Rig Operations": ["rig operation", "drilling rig"],
    "Material Handling": ["material handling", "moving materials"],
    "Excavation": ["excavation", "excavating"],
    "Trenching": ["trench", "trenching"],
    "Scaffolding": ["scaffold", "scaffolding"],
    "Roof Work": ["roof work", "on roof"],
    "Tank Cleaning": ["tank cleaning", "cleaning tank"],
    "Chemical Handling": ["chemical handling", "handling chemicals", "chemical transfer"],
    "Process Operations": ["process operation", "process unit"],
    "Plant Maintenance": ["plant maintenance"],
    "Construction": ["construction"],
    "Demolition": ["demolition"],
    "Road Work": ["road work", "road construction"],
    "Manual Handling": ["manual handling", "manual lifting"],
    "Laboratory Work": ["laboratory", "lab work"],
    "Inspection": ["inspection", "inspecting"],
    "Testing": ["testing", "test activity"],
    "Commissioning": ["commissioning"],
    "Shutdown / Turnaround": ["shutdown", "turnaround"],
}

LSR_PATTERNS = {
    "Bypassing Safety Controls": [
        "interlock bypass", "interlock defeated", "safety trip defeated",
        "alarm override", "safety control bypass",
    ],
    "Confined Space": [
        "confined space", "tank entry", "vessel entry", "manhole entry",
    ],
    "Driving": [
        "driving", "driver", "vehicle", "seatbelt", "seat belt",
    ],
    "Energy Isolation": [
        "loto", "lockout", "tagout", "energy isolation", "zero energy",
    ],
    "Hot Work": [
        "hot work", "welding", "gas cutting", "ignition source",
    ],
    "Line of Fire": [
        "line of fire", "snap-back", "moving equipment", "exclusion zone",
        "pressurized line", "dropped object",
    ],
    "Safe Mechanical Lifting": [
        "crane", "lifting", "rigging", "suspended load",
    ],
    "Work Authorisation": [
        "work permit", "permit approval", "permit conditions",
        "authorization", "authorisation",
    ],
    "Working at Height": [
        "working at height", "scaffold", "fall protection", "guardrail",
        "open edge",
    ],
}

# Additional phrase variants for free-text reports. These are deterministic
# rules, not learned labels, and should be reviewed/extended with HSE experts.
BARRIER_PHRASE_AUGMENTS = {
    "Zero energy not verified": [
        "before the isolation status had been independently confirmed",
        "without verifying isolation",
        "isolation had not been independently confirmed",
        "zero energy was not confirmed",
    ],
    "Pressurized line opened unsafely": [
        "opened a process line before",
        "opened a process connection before",
        "residual liquid escaped",
        "pressure was noticed at the joint",
        "assumed it had drained",
    ],
    "Reversing without spotter": [
        "no spotter was present",
        "without a banksman",
        "without a spotter",
    ],
    "Person inside exclusion zone": [
        "worker was still inside the swing radius",
        "worker remained inside the potential release path",
        "personnel were passing underneath",
    ],
    "Exclusion zone not established": [
        "exclusion zone was not fully established",
        "area was not cleared",
        "route was not cleared",
        "boundary was not being actively controlled",
    ],
    "Load not secured": [
        "load had not been secured",
        "load was not secured against movement",
        "sling with visible damage",
        "damaged lifting sling",
    ],
    "Lifting plan absent": [
        "before the lifting plan was available",
        "lifting plan was not available",
    ],
    "Gas test not completed": [
        "gas test had not been recorded",
        "gas test had not been completed",
        "required gas test had not been completed",
    ],
    "Hot work permit missing": [
        "started cutting before the hot work permit",
        "hot-work permit was not completed",
    ],
    "Fall protection not connected": [
        "without connecting the available fall-arrest system",
        "without using the available fall-arrest system",
    ],
    "Guardrail missing": [
        "guardrail was missing",
        "guardrail was not present",
        "one section of the guardrail missing",
    ],
    "Open edge not protected": [
        "open edge had no effective protection",
        "open edge was not protected",
    ],
    "Unsafe access": [
        "access route was incomplete",
        "designated access platform",
    ],
    "Tools/materials not secured against falling": [
        "tools were not secured",
        "materials were not secured",
    ],
    "Safety interlock bypassed": [
        "bypass was not authorised",
        "bypass was not authorized",
    ],
    "Permit expired": [
        "permit had expired",
    ],
    "Scope changed without reauthorization": [
        "scope was changed without authorization",
        "scope was changed without reauthorization",
    ],
}

ACTIVITY_PHRASE_AUGMENTS = {
    "Line Breaking": ["flange", "process connection", "line break"],
    "Pressure Testing": ["pressure test", "pressure testing"],
    "Crane / Lifting": ["suspended load", "lift supervisor", "lifting plan", "crane"],
    "Working at Height": ["temporary platform", "open edge", "fall-arrest", "fall arrest"],
    "Scaffolding": ["scaffold", "scaffolding", "guardrail"],
    "Electrical Work": ["electrical cabinet", "electrical panel", "incoming supply"],
    "Vehicle Movement": ["reversing", "banksman", "spotter", "pedestrians"],
    "Confined Space Entry": ["vessel", "confined pit", "vessel entry"],
    "Hot Work": ["cutting", "welding", "hot-work"],
    "Mechanical Maintenance": ["mechanical maintenance", "maintenance crew"],
    "Chemical Handling": ["chemical transfer", "chemical hose"],
}

LSR_PHRASE_AUGMENTS = {
    "Energy Isolation": ["isolation", "zero energy", "incoming supply", "stored pressure"],
    "Line of Fire": ["release path", "swing radius", "passing underneath", "moving equipment"],
    "Safe Mechanical Lifting": ["sling", "load", "lift supervisor", "lifting plan", "suspended load"],
    "Working at Height": ["temporary platform", "open edge", "fall-arrest", "fall arrest", "guardrail"],
    "Work Authorisation": ["permit", "authorisation", "authorization"],
}

ACTIVITY_SCORE = {
    "Confined Space Entry": 100,
    "Working at Height": 95,
    "Crane / Lifting": 95,
    "Drilling": 90,
    "Well Intervention": 90,
    "Well Servicing": 90,
    "Workover": 90,
    "Rig Operations": 90,
    "Energy Isolation / LOTO": 90,
    "Hot Work": 85,
    "Line Breaking": 85,
    "Pressure Testing": 85,
    "Electrical Work": 85,
    "Vehicle Movement": 80,
    "Driving": 80,
    "Excavation": 80,
    "Trenching": 80,
    "Chemical Handling": 75,
    "Mechanical Maintenance": 70,
    "Construction": 70,
    "Other": 30,
}

CRITICAL_BARRIERS = {
    "Safety interlock bypassed",
    "Gas testing absent",
    "Continuous gas monitoring absent",
    "LOTO not applied",
    "Zero energy not verified",
    "Hot work permit missing",
    "Person inside exclusion zone",
    "Snap-back zone not established",
    "Pressurized line opened unsafely",
    "Lifting plan absent",
    "Load capacity exceeded",
    "Fall protection not connected",
    "Open edge not protected",
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text


def clean_for_model(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(x: Any) -> Optional[float]:
    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def token_lengths(texts: list[str], tokenizer) -> np.ndarray:
    lengths = []
    for text in texts:
        ids = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        lengths.append(len(ids))
    return np.array(lengths, dtype=int)


def length_stats(lengths: np.ndarray) -> dict[str, Any]:
    if len(lengths) == 0:
        return {
            "median": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "max": 0,
        }
    return {
        "median": float(np.median(lengths)),
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(np.max(lengths)),
    }


# ============================================================
# DATASET VALIDATION / REPORTING
# ============================================================

def convert_label(value: Any) -> int:
    if isinstance(value, str):
        v = value.strip().lower()
        mapping = {
            "1": 1, "true": 1, "yes": 1, "sif": 1, "positive": 1,
            "0": 0, "false": 0, "no": 0, "non-sif": 0, "negative": 0,
        }
        if v in mapping:
            return mapping[v]
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0):
        return int(value)
    raise ValueError(f"Invalid sif_label value: {value!r}")


def validate_dataset(
    df: pd.DataFrame,
    tokenizer=None,
    near_empty_chars: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"report_text", "sif_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Required columns are: report_text, sif_label."
        )

    work = df.copy()

    work["report_text"] = work["report_text"].fillna("").astype(str)
    missing_values = {
        column: int(work[column].isna().sum())
        for column in work.columns
    }

    invalid = []
    converted = []
    for value in work["sif_label"].tolist():
        try:
            converted.append(convert_label(value))
        except ValueError:
            invalid.append(value)

    if invalid:
        examples = invalid[:10]
        raise ValueError(
            f"Invalid sif_label values found: {examples}. "
            "Labels must represent 0/1."
        )

    work["sif_label"] = np.array(converted, dtype=int)

    exact_empty = work["report_text"].str.strip().eq("")
    near_empty = (
        work["report_text"].str.strip().str.len().between(1, near_empty_chars)
    )

    duplicates = int(work["report_text"].duplicated(keep=False).sum())

    class_counts = work["sif_label"].value_counts().reindex(
        [0, 1], fill_value=0
    )
    total = len(work)
    sif_count = int(class_counts[1])
    non_sif_count = int(class_counts[0])
    sif_pct = 100 * sif_count / total if total else 0

    imbalance_ratio = (
        max(sif_count, non_sif_count) / max(1, min(sif_count, non_sif_count))
    )

    optional_metadata = [
        c for c in work.columns
        if c not in {"report_text", "sif_label"}
    ]

    stats = {
        "dataset_size": total,
        "sif_count": sif_count,
        "non_sif_count": non_sif_count,
        "sif_percentage": sif_pct,
        "non_sif_percentage": 100 - sif_pct,
        "duplicate_report_rows": duplicates,
        "empty_reports": int(exact_empty.sum()),
        "near_empty_reports": int(near_empty.sum()),
        "missing_values": missing_values,
        "optional_metadata_columns": optional_metadata,
        "class_imbalance_ratio": imbalance_ratio,
    }

    if tokenizer is not None:
        usable_texts = work.loc[~exact_empty, "report_text"].tolist()
        lengths = token_lengths(usable_texts, tokenizer)
        stats["token_length"] = length_stats(lengths)
        stats["longer_than_max_length"] = int(
            np.sum(lengths > tokenizer.model_max_length)
        )

    LOGGER.info("DATASET REPORT")
    LOGGER.info("Dataset size: %d", total)
    LOGGER.info("SIF: %d", sif_count)
    LOGGER.info("NON-SIF: %d", non_sif_count)
    LOGGER.info("SIF percentage: %.2f%%", sif_pct)
    LOGGER.info("Duplicate report rows: %d", duplicates)
    LOGGER.info("Empty reports: %d", int(exact_empty.sum()))
    LOGGER.info("Near-empty reports (<= %d chars): %d",
                near_empty_chars, int(near_empty.sum()))
    LOGGER.info("Optional metadata columns: %s",
                optional_metadata if optional_metadata else "none")
    LOGGER.info("Class imbalance ratio: %.2f", imbalance_ratio)

    if imbalance_ratio >= 4:
        LOGGER.warning(
            "Severe class imbalance detected. Training will use class-weighted loss "
            "calculated from TRAINING data only."
        )
    elif imbalance_ratio >= 2:
        LOGGER.warning("Moderate class imbalance detected.")

    if duplicates:
        LOGGER.warning(
            "Duplicate reports exist. Duplicates can cause leakage if similar/identical "
            "events cross splits. Consider --drop-duplicates or a group identifier."
        )

    if exact_empty.any():
        LOGGER.warning(
            "%d completely empty reports will be excluded from model training/evaluation "
            "because there is no text signal. The original CSV is not modified.",
            int(exact_empty.sum()),
        )
        work = work.loc[~exact_empty].copy()

    if work["sif_label"].nunique() < 2:
        raise ValueError("Both SIF and NON-SIF classes are required after cleaning.")

    return work, stats


def split_data(
    df: pd.DataFrame,
    strategy: str,
    group_column: Optional[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if strategy not in {"stratified", "group"}:
        raise ValueError("SPLIT_STRATEGY must be 'stratified' or 'group'.")

    if strategy == "group":
        if not group_column or group_column not in df.columns:
            LOGGER.warning(
                "Group split requested but no valid group column exists. "
                "Falling back to stratified splitting; leakage cannot be ruled out."
            )
            strategy = "stratified"

    if strategy == "group":
        groups = df[group_column].fillna("__MISSING_GROUP__").astype(str)

        # First hold out ~15% for test.
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=0.15, random_state=seed
        )
        train_val_idx, test_idx = next(
            gss_test.split(df, y=df["sif_label"], groups=groups)
        )

        train_val = df.iloc[train_val_idx].copy()
        test = df.iloc[test_idx].copy()

        train_val_groups = groups.iloc[train_val_idx]

        # 15% of total / 85% remaining ≈ 17.65% of train+val.
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=(0.15 / 0.85), random_state=seed + 1
        )
        train_idx_rel, val_idx_rel = next(
            gss_val.split(
                train_val,
                y=train_val["sif_label"],
                groups=train_val_groups,
            )
        )

        train = train_val.iloc[train_idx_rel].copy()
        val = train_val.iloc[val_idx_rel].copy()

        LOGGER.info(
            "GROUP split used with group column '%s'. "
            "Exact 70/15/15 proportions may vary because groups are indivisible.",
            group_column,
        )
    else:
        LOGGER.warning(
            "No group-based split is being used. If multiple reports come from "
            "the same incident/site/time period, leakage cannot be ruled out."
        )
        train, temp = train_test_split(
            df,
            test_size=0.30,
            random_state=seed,
            stratify=df["sif_label"],
        )
        val, test = train_test_split(
            temp,
            test_size=0.50,
            random_state=seed,
            stratify=temp["sif_label"],
        )

    for name, part in [("train", train), ("validation", val), ("test", test)]:
        if part["sif_label"].nunique() < 2:
            LOGGER.warning(
                "%s split contains only one class. Some metrics such as ROC-AUC "
                "may be unavailable.",
                name,
            )

    LOGGER.info(
        "Split sizes — train=%d, validation=%d, test=%d",
        len(train), len(val), len(test),
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ============================================================
# TOKENIZATION
# ============================================================

def make_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    """
    Uses overflowing token chunks instead of silently dropping the end of a
    long report. Each chunk receives the source report's label.

    At inference, chunk-level probabilities are aggregated back to one report.
    """
    base = Dataset.from_pandas(
        df[["report_text", "sif_label"]],
        preserve_index=False,
    )

    def tokenize_batch(batch):
        encoded = tokenizer(
            [clean_for_model(x) for x in batch["report_text"]],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_overflowing_tokens=True,
        )
        mapping = encoded.pop("overflow_to_sample_mapping")
        encoded["labels"] = [
            int(batch["sif_label"][i]) for i in mapping
        ]
        return encoded

    tokenized = base.map(
        tokenize_batch,
        batched=True,
        remove_columns=base.column_names,
        desc="Tokenizing reports",
    )
    return tokenized


# ============================================================
# CLASS-WEIGHTED TRAINER
# ============================================================

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        else:
            weights = self.class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(
                logits, labels, weight=weights
            )

        return (loss, outputs) if return_outputs else loss


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    counts = np.bincount(
        train_df["sif_label"].astype(int).values,
        minlength=2,
    ).astype(float)

    if np.any(counts == 0):
        raise ValueError(
            "Training split must contain both classes to calculate class weights."
        )

    # Balanced inverse-frequency weighting:
    # n_samples / (n_classes * class_count).
    n = counts.sum()
    weights = n / (2.0 * counts)

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# METRICS / THRESHOLD / CALIBRATION
# ============================================================

def binary_metrics(y_true, probabilities, threshold=0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    predictions = (probabilities >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_true, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, predictions, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None

    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "specificity": safe_float(specificity),
        "false_positive_rate": safe_float(fpr),
        "false_negative_rate": safe_float(fnr),
        "confusion_matrix": cm.tolist(),
    }

    # ROC-AUC is undefined when only one class is present.
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        result["pr_auc_average_precision"] = float(
            average_precision_score(y_true, probabilities)
        )
    else:
        result["roc_auc"] = None
        result["pr_auc_average_precision"] = None

    return result


def select_threshold(
    y_true,
    probabilities,
    criterion: str = "f1",
) -> tuple[float, list[dict[str, Any]]]:
    rows = []

    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        metrics = binary_metrics(y_true, probabilities, float(threshold))
        rows.append(metrics)

    if criterion == "f1":
        # Tie-break toward higher recall, then lower threshold.
        best = max(
            rows,
            key=lambda x: (
                x["f1"],
                x["recall"],
                -x["threshold"],
            ),
        )
    elif criterion == "macro_f1":
        best = max(
            rows,
            key=lambda x: (
                x["macro_f1"],
                x["recall"],
                -x["threshold"],
            ),
        )
    else:
        raise ValueError("Unsupported threshold criterion.")

    return float(best["threshold"]), rows


class ProbabilityCalibrator:
    """
    Optional Platt-style calibration fitted on validation probabilities.

    IMPORTANT:
        Calibration is optional and OFF by default.
        When enabled, validation data are used for calibration and threshold
        selection. The test set remains untouched.
    """

    def __init__(self):
        self.model = LogisticRegression(
            solver="lbfgs",
            random_state=DEFAULT_SEED,
        )

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw_probabilities, y_true):
        self.model.fit(self._logit(raw_probabilities), y_true)
        return self

    def transform(self, raw_probabilities):
        return self.model.predict_proba(
            self._logit(raw_probabilities)
        )[:, 1]

    def save(self, path: Path):
        payload = {
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        obj = cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        obj.model.coef_ = np.asarray(payload["coef"], dtype=float)
        obj.model.intercept_ = np.asarray(payload["intercept"], dtype=float)
        obj.model.classes_ = np.array([0, 1])
        return obj


def calibration_metrics(y_true, probabilities) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float)

    result = {
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }

    if len(np.unique(y_true)) == 2 and len(y_true) >= 10:
        fraction_pos, mean_pred = calibration_curve(
            y_true,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )
        ece = 0.0
        bins = []
        # Reconstruct quantile bins approximately from probability ordering.
        order = np.argsort(probabilities)
        chunks = np.array_split(order, min(10, len(order)))
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            avg_conf = float(np.mean(probabilities[chunk]))
            avg_acc = float(np.mean(y_true[chunk]))
            ece += (len(chunk) / len(y_true)) * abs(avg_conf - avg_acc)
            bins.append({
                "count": int(len(chunk)),
                "mean_predicted": avg_conf,
                "fraction_positive": avg_acc,
            })
        result["expected_calibration_error"] = float(ece)
        result["calibration_bins"] = bins
    else:
        result["expected_calibration_error"] = None
        result["calibration_bins"] = []

    return result


def predict_probabilities(
    trainer: Trainer,
    dataset: Dataset,
    original_row_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict every token chunk. Aggregate chunks belonging to the same source
    report by mean probability.

    Dataset creation preserves the source-row order while expanding overflow
    chunks, so overflow chunks can be assigned using the same tokenizer mapping.
    To keep this robust, the function reconstructs the mapping from the source
    report lengths.
    """
    # Rebuild the mapping deterministically using the same tokenizer behavior.
    # The tokenized dataset does not retain the mapping after remove_columns.
    # Therefore, infer chunk counts by re-tokenizing each source report.
    # This is cheap relative to Transformer inference and makes aggregation exact.
    # The caller provides a special attribute with chunk counts.
    chunk_counts = getattr(dataset, "_sif_chunk_counts", None)
    if chunk_counts is None:
        raise RuntimeError("Chunk mapping missing from dataset.")

    prediction = trainer.predict(dataset)
    logits = prediction.predictions
    chunk_probs = torch.softmax(
        torch.tensor(logits, dtype=torch.float32), dim=-1
    )[:, 1].numpy()

    report_probs = []
    pos = 0
    for count in chunk_counts:
        if count <= 0:
            report_probs.append(0.0)
        else:
            report_probs.append(float(np.mean(chunk_probs[pos:pos + count])))
            pos += count

    if pos != len(chunk_probs):
        raise RuntimeError("Chunk aggregation mismatch.")

    return np.asarray(report_probs), chunk_probs


def make_dataset_with_mapping(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
) -> Dataset:
    dataset = make_dataset(df, tokenizer, max_length)

    counts = []
    for text in df["report_text"].tolist():
        encoded = tokenizer(
            clean_for_model(text),
            truncation=True,
            max_length=max_length,
            padding=False,
            return_overflowing_tokens=True,
        )
        mapping = encoded.get("overflow_to_sample_mapping", [0])
        counts.append(len(mapping))

    dataset._sif_chunk_counts = counts
    return dataset


# ============================================================
# TRAINING / ARTIFACTS
# ============================================================

def get_training_args(
    output_dir,
    learning_rate,
    batch_size,
    epochs,
    weight_decay,
    warmup_ratio,
    seed,
    train_examples=None,
):
    # Transformers 5.x removed `warmup_ratio` from TrainingArguments.
    # Keep the user-facing CLI option for compatibility, but convert it to
    # an integer warmup_steps value before constructing TrainingArguments.
    # This is computed from the training split only.
    if train_examples is None:
        raise ValueError("train_examples is required to compute warmup_steps.")

    steps_per_epoch = max(1, math.ceil(train_examples / batch_size))
    total_steps = max(1, math.ceil(steps_per_epoch * epochs))
    warmup_steps = max(0, int(round(total_steps * float(warmup_ratio))))

    common = dict(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        data_seed=seed,
        fp16=torch.cuda.is_available(),
    )

    # transformers versions differ in the spelling of this argument.
    import inspect
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        common["eval_strategy"] = "epoch"
    else:
        common["evaluation_strategy"] = "epoch"

    return TrainingArguments(**common)


def save_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def train_model(args):
    set_global_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{args.model}'. Supported: {sorted(SUPPORTED_MODELS)}"
        )

    LOGGER.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    raw_df = pd.read_csv(args.data)
    df, dataset_stats = validate_dataset(raw_df, tokenizer=tokenizer)

    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(
            subset=["report_text", "sif_label"],
            keep="first",
        ).reset_index(drop=True)
        LOGGER.warning(
            "Dropped %d exact duplicate report/label rows because --drop-duplicates was supplied.",
            before - len(df),
        )

    train_df, val_df, test_df = split_data(
        df,
        args.split_strategy,
        args.group_column,
        args.seed,
    )

    # Token-length diagnostics against user-selected max length.
    lengths = token_lengths(df["report_text"].tolist(), tokenizer)
    stats = length_stats(lengths)
    LOGGER.info(
        "Token lengths — median=%.0f, p75=%.0f, p90=%.0f, p95=%.0f, max=%d",
        stats["median"], stats["p75"], stats["p90"], stats["p95"], stats["max"],
    )
    if stats["p95"] > args.max_length:
        LOGGER.warning(
            "At least the 95th percentile exceeds MAX_LENGTH=%d. "
            "Long reports are chunked rather than silently truncated.",
            args.max_length,
        )
    if stats["max"] > args.max_length:
        LOGGER.warning(
            "Some reports exceed MAX_LENGTH=%d. Chunk/mean aggregation is enabled.",
            args.max_length,
        )

    train_ds = make_dataset_with_mapping(train_df, tokenizer, args.max_length)
    val_ds = make_dataset_with_mapping(val_df, tokenizer, args.max_length)
    test_ds = make_dataset_with_mapping(test_df, tokenizer, args.max_length)

    LOGGER.info("Loading sequence classifier: %s", args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "NON_SIF", 1: "SIF"},
        label2id={"NON_SIF": 0, "SIF": 1},
    )

    class_weights = compute_class_weights(train_df)
    LOGGER.info(
        "Training-only class weights: NON_SIF=%.4f, SIF=%.4f",
        float(class_weights[0]),
        float(class_weights[1]),
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    training_args = get_training_args(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        train_examples=len(train_ds),
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # Transformers 5.x renamed Trainer's tokenizer argument to
        # processing_class.
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: binary_metrics(
            p.label_ids,
            torch.softmax(torch.tensor(p.predictions), dim=-1)[:, 1].numpy(),
            0.5,
        ),
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    LOGGER.info("Starting training...")
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # --------------------------------------------------------
    # Validation threshold selection — TEST NOT TOUCHED.
    # --------------------------------------------------------
    _, val_chunk_probs = predict_probabilities(
        trainer, val_ds, len(val_df)
    )
    val_probs, _ = predict_probabilities(
        trainer, val_ds, len(val_df)
    )

    if args.calibrate:
        LOGGER.warning(
            "Calibration is enabled. Calibration and threshold selection both use "
            "the validation set; the test set remains untouched."
        )
        calibrator = ProbabilityCalibrator()
        calibrator.fit(val_probs, val_df["sif_label"].values)
        val_probs_for_selection = calibrator.transform(val_probs)
        calibrator.save(output_dir / "calibrator.json")
        calibration_enabled = True
    else:
        calibrator = None
        val_probs_for_selection = val_probs
        calibration_enabled = False

    if args.threshold == "auto":
        selected_threshold, threshold_rows = select_threshold(
            val_df["sif_label"].values,
            val_probs_for_selection,
            args.threshold_criterion,
        )
    else:
        selected_threshold = float(args.threshold)
        threshold_rows = []

    LOGGER.info(
        "Selected classification threshold from VALIDATION only: %.2f",
        selected_threshold,
    )

    # --------------------------------------------------------
    # Final train/val/test evaluation.
    # --------------------------------------------------------
    train_probs, _ = predict_probabilities(
        trainer, train_ds, len(train_df)
    )
    val_probs_final = (
        calibrator.transform(val_probs) if calibrator else val_probs
    )
    test_probs_raw, _ = predict_probabilities(
        trainer, test_ds, len(test_df)
    )
    test_probs_final = (
        calibrator.transform(test_probs_raw)
        if calibrator else test_probs_raw
    )

    train_metrics = binary_metrics(
        train_df["sif_label"].values,
        train_probs,
        selected_threshold,
    )
    val_metrics = binary_metrics(
        val_df["sif_label"].values,
        val_probs_final,
        selected_threshold,
    )
    test_metrics = binary_metrics(
        test_df["sif_label"].values,
        test_probs_final,
        selected_threshold,
    )

    calibration_info = calibration_metrics(
        test_df["sif_label"].values,
        test_probs_final,
    ) if calibration_enabled else None

    # This calibration metric is only descriptive if calibration is disabled.
    # It is NOT used for model selection.
    raw_test_calibration = calibration_metrics(
        test_df["sif_label"].values,
        test_probs_raw,
    )

    train_f1 = train_metrics["f1"]
    val_f1 = val_metrics["f1"]
    test_f1 = test_metrics["f1"]
    train_loss = trainer.evaluate(
        eval_dataset=train_ds,
        metric_key_prefix="train",
    ).get("train_loss")
    val_loss = trainer.evaluate(
        eval_dataset=val_ds,
        metric_key_prefix="validation",
    ).get("validation_loss")
    test_loss = trainer.evaluate(
        eval_dataset=test_ds,
        metric_key_prefix="test",
    ).get("test_loss")

    f1_gap = train_f1 - test_f1
    loss_gap = (
        float(val_loss) - float(train_loss)
        if train_loss is not None and val_loss is not None
        else None
    )

    if f1_gap >= OVERFIT_F1_GAP or (
        loss_gap is not None and loss_gap >= OVERFIT_LOSS_GAP
    ):
        health = "OVERFITTING / POSSIBLE OVERFITTING"
        recommendation = (
            "Consider stronger regularization, fewer epochs, earlier stopping, "
            "or more diverse labelled data. Confirm with repeated/group-aware validation."
        )
    elif (
        train_f1 < UNDERFIT_F1
        and val_f1 < UNDERFIT_F1
        and test_f1 < UNDERFIT_F1
    ):
        health = "UNDERFITTING / MODEL NEEDS IMPROVEMENT"
        recommendation = (
            "Consider a stronger model, better labels, more representative data, "
            "or revised training settings."
        )
    else:
        health = "NO STRONG OVERFITTING SIGNAL UNDER CURRENT HEURISTICS"
        recommendation = (
            "This is not proof of production readiness. Review leakage, labels, "
            "class balance, subgroup performance, and HSE validity."
        )

    model_health = {
        "status": health,
        "train_test_f1_gap": f1_gap,
        "train_validation_f1_gap": train_f1 - val_f1,
        "validation_test_f1_gap": abs(val_f1 - test_f1),
        "train_loss": safe_float(train_loss),
        "validation_loss": safe_float(val_loss),
        "test_loss": safe_float(test_loss),
        "recommendation": recommendation,
    }

    # Confusion matrix / classification report on held-out test only.
    test_predictions = (test_probs_final >= selected_threshold).astype(int)
    report = classification_report(
        test_df["sif_label"].values,
        test_predictions,
        labels=[0, 1],
        target_names=["NON_SIF", "SIF"],
        zero_division=0,
        output_dict=True,
    )

    cm = np.asarray(test_metrics["confusion_matrix"])

    np.savetxt(
        output_dir / "test_confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    save_json(
        output_dir / "test_classification_report.json",
        report,
    )
    save_json(
        output_dir / "threshold_analysis_validation.json",
        threshold_rows,
    )

    # Save calibration bins if enabled.
    if calibration_enabled:
        cal_test = calibration_metrics(
            test_df["sif_label"].values,
            test_probs_final,
        )
        save_json(output_dir / "calibration_test.json", cal_test)

    metadata = {
        "timestamp_utc": now_utc(),
        "purpose": "SIF precursor detection decision-support prototype",
        "model_name": args.model,
        "seed": args.seed,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "split_strategy": args.split_strategy,
        "group_column": args.group_column,
        "threshold": selected_threshold,
        "threshold_source": (
            "validation_f1_optimization"
            if args.threshold == "auto"
            else "user_configured"
        ),
        "threshold_criterion": args.threshold_criterion,
        "calibration_enabled": calibration_enabled,
        "dataset_statistics": dataset_stats,
        "actual_split_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "class_weights_from_training_only": class_weights.tolist(),
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        },
        "model_health": model_health,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "datasets": __import__("datasets").__version__,
            "sklearn": __import__("sklearn").__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "priority_weights": DEFAULT_PRIORITY_WEIGHTS,
        "priority_weight_warning": (
            "Prototype/manual weights. They are not scientifically validated "
            "and require HSE/domain validation."
        ),
        "raw_test_calibration_if_no_calibration": raw_test_calibration,
    }
    save_json(output_dir / "model_metadata.json", metadata)

    print("\n" + "=" * 75)
    print("FINAL MODEL EVALUATION")
    print("=" * 75)
    for name, metrics in [
        ("TRAIN", train_metrics),
        ("VALIDATION", val_metrics),
        ("TEST", test_metrics),
    ]:
        print(
            f"{name:12s} | "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"BalancedAcc={metrics['balanced_accuracy']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"PR-AUC={metrics['pr_auc_average_precision']}"
        )

    print("\nSelected threshold:", selected_threshold)
    print("Model health:", health)
    print("Test confusion matrix:")
    print(cm)
    print("\nArtifacts saved in:", output_dir.resolve())

    return trainer


# ============================================================
# INFERENCE ENGINE
# ============================================================

@dataclass
class EngineConfig:
    threshold: float = 0.50
    calibration_enabled: bool = False
    max_length: int = DEFAULT_MAX_LENGTH
    model_name: str = DEFAULT_MODEL


class SIFEngine:
    def __init__(self, model_dir: str = DEFAULT_OUTPUT_DIR):
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory '{self.model_dir}' does not exist. Train first."
            )

        metadata_path = self.model_dir / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing {metadata_path}. The model directory is incomplete."
            )

        self.metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        self.config = EngineConfig(
            threshold=float(self.metadata["threshold"]),
            calibration_enabled=bool(
                self.metadata.get("calibration_enabled", False)
            ),
            max_length=int(self.metadata["max_length"]),
            model_name=self.metadata["model_name"],
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir)
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Model/tokenizer could not be loaded from '{self.model_dir}': {exc}"
            ) from exc

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        calibrator_path = self.model_dir / "calibrator.json"
        if self.config.calibration_enabled and calibrator_path.exists():
            self.calibrator = ProbabilityCalibrator.load(calibrator_path)
        else:
            self.calibrator = None
            self.config.calibration_enabled = False

    def _chunk_probabilities(self, text: str) -> list[float]:
        text = clean_for_model(text)
        if not text:
            return []

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
            return_overflowing_tokens=True,
            return_tensors=None,
        )

        input_ids = encoded["input_ids"]
        attention_masks = encoded["attention_mask"]

        probs = []
        for ids, mask in zip(input_ids, attention_masks):
            inputs = {
                "input_ids": torch.tensor([ids], dtype=torch.long).to(self.device),
                "attention_mask": torch.tensor([mask], dtype=torch.long).to(self.device),
            }

            with torch.no_grad():
                logits = self.model(**inputs).logits
                p = torch.softmax(logits, dim=-1)[0, 1].item()
            probs.append(float(p))

        return probs

    def predict_sif(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("report text cannot be empty.")

        raw_chunks = self._chunk_probabilities(text)
        if not raw_chunks:
            raise ValueError("report text produced no model tokens.")

        raw_score = float(np.mean(raw_chunks))

        if self.calibrator is not None:
            calibrated = float(
                self.calibrator.transform([raw_score])[0]
            )
        else:
            calibrated = None

        decision_score = (
            calibrated if calibrated is not None else raw_score
        )

        return {
            "prediction": (
                "SIF_PRECURSOR"
                if decision_score >= self.config.threshold
                else "NON_SIF"
            ),
            "raw_model_score": raw_score,
            "calibrated_probability": calibrated,
            "classification_threshold": self.config.threshold,
            "threshold_source": self.metadata.get(
                "threshold_source", "unknown"
            ),
            "number_of_text_chunks": len(raw_chunks),
            "probability_note": (
                "Raw model score is a model output, not a calibrated real-world "
                "probability. A calibrated probability is supplied only when "
                "calibration was enabled during training."
            ),
        }


# ============================================================
# RULE-BASED NLP
# ============================================================

NEGATION_TERMS = [
    "not", "no", "never", "without", "did not", "was not",
    "were not", "wasn't", "weren't", "completed", "complete",
    "correctly", "properly", "adequate", "available", "present",
    "in place", "followed", "applied", "verified", "connected",
    "approved", "valid", "secured", "controlled",
]


def find_evidence(text: str, phrase: str, window: int = 80) -> str:
    """
    Return only text actually present in the report around the matched phrase.
    """
    original = str(text)
    lower = normalize_text(original)
    phrase_lower = normalize_text(phrase)

    idx = lower.find(phrase_lower)
    if idx < 0:
        return ""

    start = max(0, idx - window)
    end = min(len(original), idx + len(phrase) + window)
    return original[start:end].strip()


def is_negated(text: str, phrase: str) -> bool:
    """
    Lightweight local negation check. It is deliberately conservative:
    it avoids declaring a barrier failure when a matched failure phrase is
    directly negated, but does not claim full linguistic understanding.
    """
    normalized = normalize_text(text)
    phrase_norm = normalize_text(phrase)
    idx = normalized.find(phrase_norm)

    if idx < 0:
        return False

    context = normalized[max(0, idx - 80):idx]
    patterns = [
        r"\bnot\s+(?:was\s+|were\s+|is\s+|are\s+)?$",
        r"\bno\s+(?:was\s+|were\s+|is\s+|are\s+)?$",
        r"\bwithout\s*$",
    ]

    # If the phrase itself already contains "not", don't treat that as
    # negating the failure; "LOTO not applied" is a positive failure statement.
    if "not" in phrase_norm or "without" in phrase_norm or "absent" in phrase_norm:
        return False

    return any(re.search(pattern, context) for pattern in patterns)


def extract_activities(text: str) -> list[str]:
    normalized = normalize_text(text)
    found = []

    # Match longer/more specific phrases first. Multiple activities are allowed.
    for activity, phrases in ACTIVITY_PATTERNS.items():
        candidates = list(phrases) + ACTIVITY_PHRASE_AUGMENTS.get(activity, [])
        if any(normalize_text(p) in normalized for p in candidates):
            found.append(activity)

    # A few high-confidence contextual fallbacks for free-form reports.
    contextual = [
        ("Line Breaking", ["flange", "process connection", "line section"]),
        ("Crane / Lifting", ["suspended load", "lifting sling", "lift supervisor"]),
        ("Working at Height", ["open edge", "fall-arrest", "fall arrest", "roof inspection"]),
        ("Electrical Work", ["electrical cabinet", "electrical panel", "absence of voltage"]),
        ("Vehicle Movement", ["banksman", "spotter", "reversing toward", "pedestrians were using"]),
        ("Confined Space Entry", ["confined pit", "vessel", "tank entry"]),
        ("Hot Work", ["welding", "cutting", "fire watch", "combustible"]),
        ("Mechanical Maintenance", ["mechanical maintenance", "maintenance crew"]),
        ("Chemical Handling", ["chemical transfer", "chemical hose"]),
    ]
    for activity, phrases in contextual:
        if activity not in found and any(normalize_text(p) in normalized for p in phrases):
            found.append(activity)

    return found if found else ["Other"]


def extract_life_saving_rules(text: str) -> list[str]:
    normalized = normalize_text(text)
    found = []

    for rule, phrases in LSR_PATTERNS.items():
        candidates = list(phrases) + LSR_PHRASE_AUGMENTS.get(rule, [])
        if any(normalize_text(p) in normalized for p in candidates):
            found.append(rule)

    # High-confidence contextual mappings that are common in safety reports.
    contextual = {
        "Energy Isolation": [
            "electrical cabinet", "electrical panel", "incoming supply",
            "absence of voltage", "isolated the drive", "zero pressure",
            "trapped pressure", "stored pressure", "isolation point",
        ],
        "Line of Fire": [
            "release path", "swing radius", "passing underneath",
            "moving coupling", "moving parts", "potential release",
        ],
        "Safe Mechanical Lifting": [
            "lifting sling", "damaged sling", "load was secured",
            "lifting plan", "lift supervisor", "suspended load",
        ],
        "Working at Height": [
            "temporary platform", "open edge", "fall-arrest", "fall arrest",
            "guardrail", "roof inspection",
        ],
        "Work Authorisation": [
            "permit", "work authorization", "work authorisation",
            "authorisation", "authorization",
        ],
        "Hot Work": ["welding", "cutting", "fire watch", "combustible material"],
        "Driving": ["seat belt", "seatbelt", "driver", "vehicle", "journey"],
        "Confined Space": ["confined pit", "vessel", "tank entry", "confined space"],
        "Bypassing Safety Controls": ["alarm override", "bypass", "interlock"],
    }
    for rule, phrases in contextual.items():
        if rule not in found and any(normalize_text(p) in normalized for p in phrases):
            found.append(rule)

    return found if found else ["Nill"]


def extract_barriers(text: str) -> list[str]:
    """Return barrier-failure names only; evidence is intentionally not exposed."""
    normalized = normalize_text(text)
    results = []

    for barrier, pattern_info in BARRIER_PATTERNS.items():
        positive_phrases = list(pattern_info["positive"]) + BARRIER_PHRASE_AUGMENTS.get(barrier, [])
        negative_phrases = pattern_info["negative"]

        # Explicit control-present phrases take precedence.
        if any(normalize_text(p) in normalized for p in negative_phrases):
            continue

        matched = None
        for phrase in positive_phrases:
            phrase_norm = normalize_text(phrase)
            if phrase_norm in normalized:
                matched = phrase_norm
                break

        if matched is None:
            continue
        if is_negated(text, matched):
            continue
        results.append(barrier)

    # Additional high-confidence phrases for common free-text variants.
    extra_barriers = [
        ("Vehicle-pedestrian segregation inadequate",
         ["pedestrians were using the same route", "same route", "pedestrian segregation", "shared route"]),
        ("Reversing without spotter",
         ["no spotter", "without a spotter", "without a banksman", "reversed without"]),
        ("Mobile phone use while driving",
         ["mobile phone while the vehicle was moving", "phone while driving", "reached for a mobile phone"]),
        ("LOTO not applied",
         ["incoming supply had not been isolated", "had not been isolated", "without isolating", "not isolated"]),
        ("Zero energy not verified",
         ["without verifying isolation", "zero-energy verification", "zero energy not verified",
          "zero energy had not been verified", "absence of voltage had not been verified"]),
        ("Residual energy not released",
         ["trapped pressure", "residual pressure", "residual energy", "had not been released"]),
        ("Gas test not completed",
         ["gas test had not been recorded", "gas test had not been completed",
          "required gas test had not been completed", "gas test was not completed"]),
        ("Continuous gas monitoring absent",
         ["continuous gas monitoring arrangements had not been confirmed",
          "continuous gas monitoring was not confirmed", "continuous gas monitoring not available"]),
        ("Rescue arrangement not confirmed",
         ["rescue team and continuous atmosphere monitoring arrangements had not been confirmed",
          "rescue arrangement had not been confirmed", "rescue arrangements had not been confirmed"]),
        ("Person inside exclusion zone",
         ["worker was still inside the swing radius", "worker remained inside the potential release path",
          "personnel were passing underneath", "person remained in the exclusion zone"]),
        ("Exclusion zone not established",
         ["exclusion zone was not fully established", "exclusion boundary was not",
          "boundary was not being actively controlled", "area was not cleared"]),
        ("Unsuitable rigging",
         ["damaged lifting sling", "sling with visible damage", "damaged sling", "unsuitable sling"]),
        ("Lifting equipment inspection overdue",
         ["lifting equipment inspection was overdue", "lifting equipment was not inspected"]),
        ("Unsafe access",
         ["designated access route was incomplete", "access route was incomplete",
          "without using the designated access platform", "unsafe access"]),
        ("Fall protection not connected",
         ["without using the available fall-arrest system", "without connecting the available fall-arrest system"]),
        ("Guardrail missing",
         ["one section of the guardrail missing", "guardrail was missing", "guardrail was not present"]),
        ("Open edge not protected",
         ["open edge had no effective protection", "open edge was not protected"]),
        ("Hot work permit missing",
         ["hot work permit had not been completed", "without completing the required hot work permit",
          "hot-work permit was not completed"]),
        ("Fire watch absent",
         ["without a fire watch", "fire watch was absent", "no fire watch"]),
        ("Flammables not removed",
         ["nearby combustible material was not removed", "combustible material was not removed",
          "combustible packaging was not removed"]),
        ("Permit expired", ["permit had expired", "expired permit"]),
        ("Permit missing", ["permit was missing", "no permit was available", "without a permit"]),
        ("Work started before permit approval", ["started work before permit approval", "began work before permit approval"]),
    ]
    for barrier, phrases in extra_barriers:
        if barrier not in results and any(normalize_text(p) in normalized for p in phrases):
            results.append(barrier)

    return results if results else ["Nill"]


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def extract_exposure(text: str) -> Optional[int]:
    normalized = normalize_text(text)

    numeric_patterns = [
        r"\b(\d+)\s+(?:workers?|technicians?|personnel|people|employees?|persons?|staff|operators?|drivers?)\b",
        r"\b(\d+)\s+(?:worker|workers)\s+(?:were\s+)?exposed\b",
        r"\bexposed\s+to\s+(\d+)\s+(?:workers?|people|persons?|personnel)\b",
    ]

    for pattern in numeric_patterns:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))

    word_pattern = (
        r"\b(" + "|".join(NUMBER_WORDS.keys()) +
        r")\s+(?:workers?|technicians?|personnel|people|employees?|persons?|staff|operators?|drivers?)\b"
    )
    match = re.search(word_pattern, normalized)
    if match:
        return NUMBER_WORDS[match.group(1)]

    # If no explicit exposure count is stated, the prototype convention is 0.
    # This means "no exposure count was available in the report"; it does not
    # prove that zero people were exposed.
    return 0


# ============================================================
# PRIORITY SCORE — TRANSPARENT, CONFIGURABLE, NOT VALIDATED
# ============================================================

def activity_component(activities: list[str]) -> float:
    known = [
        ACTIVITY_SCORE[a]
        for a in activities
        if a in ACTIVITY_SCORE
    ]
    return float(max(known)) if known else 30.0


def barrier_component(barriers: list[dict[str, str]]) -> float:
    if not barriers:
        return 0.0

    scores = []
    for barrier in barriers:
        scores.append(
            100.0 if barrier in CRITICAL_BARRIERS else 60.0
        )
    return float(max(scores))


def exposure_component(exposure: Optional[int]) -> Optional[float]:
    if exposure is None:
        return None
    if exposure <= 1:
        return 25.0
    if exposure <= 5:
        return 50.0
    if exposure <= 10:
        return 75.0
    return 100.0


def calculate_priority(
    sif_score: float,
    activities: list[str],
    barriers: list[dict[str, str]],
    exposure: Optional[int],
    weights: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    weights = weights or DEFAULT_PRIORITY_WEIGHTS

    sif = float(sif_score * 100)
    activity = activity_component(activities)
    barrier = barrier_component(barriers)
    exposure = exposure_component(exposure)

    if exposure is None:
        # Renormalize across known components instead of treating unknown
        # exposure as zero.
        known_weights = {
            k: v for k, v in weights.items()
            if k != "exposure"
        }
        denominator = sum(known_weights.values()) or 1.0

        score = (
            sif * known_weights["sif"] +
            activity * known_weights["activity"] +
            barrier * known_weights["barrier"]
        ) / denominator
    else:
        total_weight = sum(weights.values()) or 1.0
        score = (
            sif * weights["sif"] +
            activity * weights["activity"] +
            barrier * weights["barrier"] +
            exposure * weights["exposure"]
        ) / total_weight

    return {
        "score": round(float(score), 2),
        "components": {
            "sif": round(sif, 2),
            "activity": round(activity, 2),
            "barrier": round(barrier, 2),
            "exposure": (
                round(exposure, 2)
                if exposure is not None else None
            ),
        },
        "exposure_status": (
            "known" if exposure is not None else "unknown_not_detected"
        ),
        "weights": weights,
        "warning": (
            "Prototype/manual priority weights. These are not scientifically "
            "validated and require HSE/domain validation before operational use."
        ),
    }


def analyze_report(engine: SIFEngine, text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("report text cannot be empty.")

    if len(text) > MAX_API_TEXT_CHARS:
        raise ValueError(
            f"report text exceeds the maximum allowed length of "
            f"{MAX_API_TEXT_CHARS} characters."
        )

    sif = engine.predict_sif(text)
    activities = extract_activities(text) or ["Other"]
    rules = extract_life_saving_rules(text) or ["None"]
    barriers = extract_barriers(text) or ["None"]
    exposure = extract_exposure(text)
    if exposure is None:
        exposure = 0

    priority = calculate_priority(
        sif_score=(
            sif["calibrated_probability"]
            if sif["calibrated_probability"] is not None
            else sif["raw_model_score"]
        ),
        activities=activities,
        barriers=barriers,
        exposure=exposure,
    )

    return {
        "sif_prediction": sif["prediction"],
        "sif_score": sif["raw_model_score"],
        "activity": activities,
        "life_saving_rule": rules,
        "barrier_failure": barriers,
        "exposure_count": exposure,
        "priority_score": priority["score"],
        "decision_support_note": (
            "This output is decision support for human/HSE review. It is not a "
            "guaranteed safety conclusion and does not replace safety professionals."
        ),
    }


# ============================================================
# BATCH PREDICTION — COMPLETELY NEW CSV
# ============================================================


def _force_final_csv_values(df):
    """Guarantee stable, human-readable values in the final prediction CSV."""
    import pandas as pd

    required_defaults = {
        "activity": "Other",
        "life_saving_rule": "None",
        "barrier_failure": "None",
        "exposure_count": 0,
    }

    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            # Treat pandas NaN/NA and empty/whitespace strings as missing.
            df[col] = df[col].apply(
                lambda x: default
                if pd.isna(x) or (isinstance(x, str) and not x.strip())
                else x
            )

    # Exposure is an integer count in the final CSV.
    def _to_exposure(x):
        try:
            if pd.isna(x):
                return 0
            return max(0, int(float(x)))
        except (TypeError, ValueError):
            return 0

    df["exposure_count"] = df["exposure_count"].apply(_to_exposure)

    # Explicitly prevent accidental debug/internal columns from leaking out.
    final_columns = [
        "report_id",
        "report_text",
        "sif_prediction",
        "sif_score",
        "activity",
        "life_saving_rule",
        "barrier_failure",
        "exposure_count",
        "priority_score",
        "priority_rank",
    ]

    for col in final_columns:
        if col not in df.columns:
            if col == "report_id":
                df[col] = [f"ROW_{i+1}" for i in range(len(df))]
            elif col == "report_text":
                df[col] = ""
            elif col == "sif_prediction":
                df[col] = "NON_SIF"
            elif col == "sif_score":
                df[col] = 0.0
            elif col == "priority_score":
                df[col] = 0.0
            elif col == "priority_rank":
                df[col] = range(1, len(df) + 1)
            elif col == "activity":
                df[col] = "Other"
            elif col == "life_saving_rule":
                df[col] = "Nill"
            elif col == "barrier_failure":
                df[col] = "Nill"
            elif col == "exposure_count":
                df[col] = 0

    return df[final_columns]

def predict_csv(
    engine: SIFEngine,
    input_path: str,
    output_path: str,
):
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV not found: {input_path}")
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    if "report_text" not in df.columns:
        raise ValueError(
            "Prediction CSV must contain a 'report_text' column. "
            "It does NOT need sif_label."
        )

    if len(df) > 100_000:
        raise ValueError("Prediction CSV is too large for this prototype (max 100,000 rows).")

    outputs = []

    for idx, row in df.iterrows():
        text = "" if pd.isna(row["report_text"]) else str(row["report_text"])

        if not text.strip():
            result = {
                "prediction_status": "INVALID_EMPTY_REPORT",
                "sif_prediction": None,
                "sif_score": None,
                "activity": ["Other"],
                "life_saving_rule": ["Nill"],
                "barrier_failure": ["Nill"],
                "exposure_count": 0,
                "exposure_status": "not_available",
                "priority_score": None,
                "priority_rank": None,
            }
        else:
            try:
                result = analyze_report(engine, text)
                # Normalize all fallback values before writing the final CSV.
                result["activity"] = result.get("activity") or ["Other"]
                result["life_saving_rule"] = result.get("life_saving_rule") or ["Nill"]
                result["barrier_failure"] = result.get("barrier_failure") or ["Nill"]
                # Prefer a supplied numeric exposure_count from the input CSV.
                # Otherwise use text extraction; if nothing is available, use 0.
                if "exposure_count" in df.columns and pd.notna(row.get("exposure_count")):
                    try:
                        result["exposure_count"] = max(0, int(float(row["exposure_count"])))
                    except (TypeError, ValueError):
                        result["exposure_count"] = 0
                else:
                    result["exposure_count"] = int(result.get("exposure_count") or 0)
                # Recalculate priority using the final exposure value supplied by the row.
                priority = calculate_priority(
                    sif_score=float(result.get("sif_score") or 0.0),
                    activities=result.get("activity", ["Other"]),
                    barriers=result.get("barrier_failure", ["None"]),
                    exposure=int(result.get("exposure_count") or 0),
                )
                result["priority_score"] = priority["score"]
                result["prediction_status"] = "OK"
            except Exception as exc:
                LOGGER.error("Row %d failed: %s", idx, exc)
                result = {
                    "prediction_status": f"ERROR: {exc}",
                    "sif_prediction": None,
                    "sif_score": None,
                    "activity": ["Other"],
                    "life_saving_rule": ["Nill"],
                    "barrier_failure": ["Nill"],
                    "exposure_count": 0,
                    "exposure_status": "not_available",
                    "priority_score": None,
                    "priority_rank": None,
                }

        outputs.append(result)

    # Final, presentation-ready output. Internal/debug fields are deliberately omitted.
    def list_to_text(x):
        if isinstance(x, list):
            return " | ".join(str(v) for v in x) if x else "None"
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "None"
        return str(x)

    final = pd.DataFrame({
        "report_id": (
            df["report_id"].astype(str).tolist()
            if "report_id" in df.columns
            else [str(i + 1) for i in range(len(df))]
        ),
        "report_text": df["report_text"].fillna("").astype(str).tolist(),
        "sif_prediction": [x.get("sif_prediction") for x in outputs],
        "sif_score": [x.get("sif_score") for x in outputs],
        "activity": [list_to_text(x.get("activity", [])) for x in outputs],
        "life_saving_rule": [list_to_text(x.get("life_saving_rule", [])) for x in outputs],
        "barrier_failure": [list_to_text(x.get("barrier_failure", [])) for x in outputs],
        "exposure_count": [x.get("exposure_count") for x in outputs],
        "priority_score": [x.get("priority_score") for x in outputs],
    })

    final["_original_order"] = np.arange(len(final))
    final = final.sort_values(
        by=["priority_score", "_original_order"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    final["priority_rank"] = np.where(
        final["priority_score"].notna(),
        np.arange(1, len(final) + 1),
        np.nan,
    )
    final = final.drop(columns=["_original_order"])

    # Exact final schema requested for the SIH demo.
    final = final[[
        "report_id",
        "report_text",
        "sif_prediction",
        "sif_score",
        "activity",
        "life_saving_rule",
        "barrier_failure",
        "exposure_count",
        "priority_score",
        "priority_rank",
    ]]

    final.to_csv(output_path, index=False)

    LOGGER.info(
        "Predicted %d new reports. Results saved to %s",
        len(final),
        output_path,
    )


# ============================================================
# FASTAPI
# ============================================================

def create_api(model_dir: str):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI dependencies are missing. Install requirements.txt."
        ) from exc

    app = FastAPI(
        title="SIH 26165 SIF Precursor Detection API",
        version="2.0",
        description=(
            "Decision-support prototype for SIF precursor detection, safety "
            "factor extraction and transparent event prioritization."
        ),
    )

    try:
        engine = SIFEngine(model_dir)
        model_loaded = True
        load_error = None
    except Exception as exc:
        LOGGER.exception("Model loading failed.")
        engine = None
        model_loaded = False
        load_error = str(exc)

    cors_origins = [
        x.strip()
        for x in os.getenv("CORS_ORIGINS", "").split(",")
        if x.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    class ReportRequest(BaseModel):
        report: str = Field(
            min_length=1,
            max_length=MAX_API_TEXT_CHARS,
        )

    class RankRequest(BaseModel):
        reports: list[str] = Field(
            min_length=1,
            max_length=MAX_API_BATCH,
        )

    @app.get("/health")
    def health():
        return {
            "api_running": True,
            "model_loaded": model_loaded,
            "model_name": (
                engine.config.model_name if engine else None
            ),
            "device": (
                str(engine.device) if engine else None
            ),
            "load_error": load_error,
        }

    @app.get("/")
    def root():
        return {
            "project": "SIH 26165",
            "service": "SIF Precursor Detection",
            "model_loaded": model_loaded,
            "decision_support": True,
        }

    @app.post("/predict")
    def predict(request: ReportRequest):
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded.",
            )
        try:
            return analyze_report(engine, request.report)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception:
            LOGGER.exception("Prediction failed.")
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. Check server logs.",
            )

    @app.post("/rank")
    def rank(request: RankRequest):
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded.",
            )

        results = []
        for text in request.reports:
            try:
                results.append(analyze_report(engine, text))
            except ValueError as exc:
                results.append({
                    "prediction_status": "INVALID",
                    "error": str(exc),
                })
            except Exception:
                LOGGER.exception("One /rank item failed.")
                results.append({
                    "prediction_status": "ERROR",
                    "error": "Prediction failed.",
                })

        valid = [
            r for r in results
            if r.get("priority_score") is not None
        ]
        invalid = [
            r for r in results
            if r not in valid
        ]

        valid.sort(
            key=lambda x: x["priority_score"],
            reverse=True,
        )

        for rank_number, item in enumerate(valid, start=1):
            item["priority_rank"] = rank_number

        return {
            "ranked_events": valid + invalid,
            "decision_support_note": (
                "Ranking uses prototype/manual priority weights and requires "
                "HSE/domain validation."
            ),
        }

    return app


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="SIH 26165 SIF precursor detection engine"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "predict", "predict-csv", "api"],
    )

    parser.add_argument("--data", help="Training CSV or new prediction CSV.")
    parser.add_argument("--text", help="One new safety report.")
    parser.add_argument("--output", default="predictions.csv")

    parser.add_argument(
        "--model",
        default=os.getenv("SIF_MODEL", DEFAULT_MODEL),
        choices=sorted(SUPPORTED_MODELS),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--split-strategy",
        choices=["stratified", "group"],
        default="stratified",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        help="Column such as incident_id/site_id for leakage-aware group splitting.",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop exact duplicate report+label rows before splitting.",
    )

    parser.add_argument(
        "--threshold",
        default="auto",
        help="Use 'auto' for validation F1 optimization or supply e.g. 0.50.",
    )
    parser.add_argument(
        "--threshold-criterion",
        choices=["f1", "macro_f1"],
        default="f1",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Fit optional probability calibration on validation data.",
    )

    return parser



def _force_final_csv_values(df):
    """Final hard guarantee for requested CSV fallback values and schema."""
    import pandas as pd

    # Normalize every missing/blank value in the requested categorical fields.
    for col, fallback in (
        ("activity", "Other"),
        ("life_saving_rule", "Nill"),
        ("barrier_failure", "Nill"),
    ):
        if col not in df.columns:
            df[col] = fallback
        else:
            df[col] = df[col].astype("object")
            df[col] = df[col].where(df[col].notna(), fallback)
            df[col] = df[col].replace(r"^\s*$", fallback, regex=True)
            # Also catch string representations produced by intermediate code.
            df.loc[df[col].astype(str).str.strip().str.lower().isin(
                ["nan", "none", "null", "<na>"]
            ) & df[col].isna(), col] = fallback

    if "exposure_count" not in df.columns:
        df["exposure_count"] = 0
    else:
        df["exposure_count"] = pd.to_numeric(
            df["exposure_count"], errors="coerce"
        ).fillna(0).clip(lower=0).astype(int)

    final_columns = [
        "report_id",
        "report_text",
        "sif_prediction",
        "sif_score",
        "activity",
        "life_saving_rule",
        "barrier_failure",
        "exposure_count",
        "priority_score",
        "priority_rank",
    ]

    for col in final_columns:
        if col not in df.columns:
            if col == "activity":
                df[col] = "Other"
            elif col in ("life_saving_rule", "barrier_failure"):
                df[col] = "Nill"
            elif col == "exposure_count":
                df[col] = 0
            elif col == "sif_prediction":
                df[col] = "NON_SIF"
            elif col in ("sif_score", "priority_score"):
                df[col] = 0.0
            elif col == "priority_rank":
                df[col] = range(1, len(df) + 1)
            else:
                df[col] = ""

    return df[final_columns]

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        if not args.data:
            parser.error("--data is required in train mode.")
        if not os.path.exists(args.data):
            parser.error(f"Training CSV not found: {args.data}")
        if args.max_length < 32:
            parser.error("--max-length should be at least 32.")
        train_model(args)
        return

    if args.mode == "predict":
        if not args.text:
            parser.error("--text is required in predict mode.")
        try:
            engine = SIFEngine(args.output_dir)
            result = analyze_report(engine, args.text)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as exc:
            LOGGER.error("%s", exc)
            sys.exit(1)
        return

    if args.mode == "predict-csv":
        if not args.data:
            parser.error("--data is required in predict-csv mode.")
        try:
            engine = SIFEngine(args.output_dir)
            predict_csv(engine, args.data, args.output)
        except Exception as exc:
            LOGGER.error("%s", exc)
            sys.exit(1)
        return

    if args.mode == "api":
        try:
            import uvicorn
            app = create_api(args.output_dir)
            uvicorn.run(app, host="127.0.0.1", port=8000)
        except Exception as exc:
            LOGGER.error("%s", exc)
            sys.exit(1)
        return


if __name__ == "__main__":
    main()
