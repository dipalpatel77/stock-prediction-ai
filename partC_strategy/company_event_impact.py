from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Iterable
import pandas as pd
import numpy as np

# Optional sentiment helper
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

@dataclass
class CompanyEvent:
    id: str
    date: pd.Timestamp
    headline: str
    event_type: str           # e.g., 'product_launch','bankruptcy','expansion','earnings','guidance','m_and_a','lawsuit','recall','ceo_change'
    expected_direction: str  # 'POSITIVE'|'NEGATIVE'|'MIXED'
    severity: float = 1.0     # 0..3
    confidence: float = 0.5   # 0..1
    effective_date: Optional[pd.Timestamp] = None
    notes: Optional[str] = None

class CompanyEventImpactModel:
    """
    Detect and apply company-specific event impacts:
      - detect events from headlines or load CSV
      - estimate signed impact score
      - apply adjustments to signals DataFrame (Company_Event_Score, Company_Adjustment, Company_Expected_Direction, Company_Confidence)
      - optionally scale Position_Size_Pct and adjust SL/TP
    """
    DEFAULT_SIGNATURES = {
        "product_launch": {"dir": "POSITIVE", "base": 0.5},
        "bankruptcy": {"dir": "NEGATIVE", "base": -2.0},
        "expansion": {"dir": "POSITIVE", "base": 0.6},
        "earnings_beat": {"dir": "POSITIVE", "base": 1.0},
        "earnings_miss": {"dir": "NEGATIVE", "base": -1.0},
        "guidance_up": {"dir": "POSITIVE", "base": 0.8},
        "guidance_down": {"dir": "NEGATIVE", "base": -0.8},
        "m_and_a": {"dir": "MIXED", "base": 0.2},
        "lawsuit": {"dir": "NEGATIVE", "base": -0.7},
        "recall": {"dir": "NEGATIVE", "base": -0.9},
        "ceo_change": {"dir": "MIXED", "base": 0.0},
        "other": {"dir": "MIXED", "base": 0.0}
    }

    def __init__(self, clamp_min: float = 0.05, clamp_max: float = 3.0):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def load_events_from_csv(self, path: str, date_col: str = "date") -> List[CompanyEvent]:
        df = pd.read_csv(path)
        events: List[CompanyEvent] = []
        for i, r in df.iterrows():
            try:
                d = pd.to_datetime(r.get(date_col))
                ev = CompanyEvent(
                    id=str(r.get("id", f"csv_{i}")),
                    date=d,
                    headline=str(r.get("headline", "")),
                    event_type=str(r.get("event_type", "other")),
                    expected_direction=str(r.get("expected_direction", "MIXED")).upper(),
                    severity=float(r.get("severity", 1.0)) if pd.notna(r.get("severity", None)) else 1.0,
                    confidence=float(r.get("confidence", 0.5)) if pd.notna(r.get("confidence", None)) else 0.5,
                    effective_date=pd.to_datetime(r.get("effective_date")) if pd.notna(r.get("effective_date", None)) else None,
                    notes=str(r.get("notes", ""))
                )
                events.append(ev)
            except Exception:
                continue
        return events

    def detect_events_from_headlines(
        self,
        headlines: Iterable[Dict[str, Any]],
        keywords_map: Optional[Dict[str, List[str]]] = None,
        date_key: str = "date",
        text_key: str = "headline"
    ) -> List[CompanyEvent]:
        if keywords_map is None:
            keywords_map = {
                "product_launch": ["launch", "introduce", "releases", "debut", "announces new product"],
                "bankruptcy": ["bankrupt", "bankruptcy", "insolvency", "filed for bankruptcy", "npa"],
                "expansion": ["expansion", "opens", "new plant", "capacity", "acquires facility", "investment"],
                "earnings_beat": ["beats estimates", "beats expectations", "earnings beat", "beat eps", "surpasses estimates"],
                "earnings_miss": ["misses estimates", "missed expectations", "earnings miss", "miss eps"],
                "guidance_up": ["raises guidance", "upgrades guidance", "increases outlook"],
                "guidance_down": ["lowers guidance", "cuts guidance", "reduces outlook"],
                "m_and_a": ["acquire", "merger", "m&a", "takeover", "buyout"],
                "lawsuit": ["sued", "lawsuit", "settlement", "investigation"],
                "recall": ["recall", "withdraw product", "safety issue"],
                "ceo_change": ["ceo", "chief executive", "appoints new", "resigns", "stepping down"]
            }
        events: List[CompanyEvent] = []
        for i, item in enumerate(headlines):
            text = str(item.get(text_key, "")).lower()
            date = pd.to_datetime(item.get(date_key, datetime.utcnow()))
            matched = False
            for etype, kws in keywords_map.items():
                for kw in kws:
                    if kw.lower() in text:
                        sig = self.DEFAULT_SIGNATURES.get(etype, self.DEFAULT_SIGNATURES["other"])
                        expected = sig["dir"]
                        # crude sentiment with VADER if available to refine expected/confidence
                        confidence = 0.6
                        severity = 1.0
                        if _VADER:
                            vs = _VADER.polarity_scores(str(item.get(text_key, "")))
                            # stronger compound -> higher confidence and adjust expected
                            comp = vs.get("compound", 0.0)
                            confidence = min(1.0, 0.4 + abs(comp))
                            if comp > 0.4:
                                expected = "POSITIVE"
                            elif comp < -0.4:
                                expected = "NEGATIVE"
                            severity = 1.0 + min(2.0, abs(comp) * 3.0)
                        ev = CompanyEvent(
                            id=f"det_{i}_{etype}",
                            date=pd.to_datetime(date),
                            headline=item.get(text_key, ""),
                            event_type=etype,
                            expected_direction=expected,
                            severity=severity,
                            confidence=confidence,
                            effective_date=None,
                            notes=f"keyword:{kw}"
                        )
                        events.append(ev)
                        matched = True
                        break
                if matched:
                    break
        return events

    def estimate_event_impact(self, event: CompanyEvent, sector_bias: float = 0.0) -> float:
        sig = self.DEFAULT_SIGNATURES.get(event.event_type, self.DEFAULT_SIGNATURES["other"])
        base = float(sig.get("base", 0.0))
        impact = base * float(event.severity) * float(event.confidence) * (1.0 + float(sector_bias))
        # small floor
        if abs(impact) < 1e-6:
            impact = 0.0
        return float(impact)

    def apply_event_adjustments(
        self,
        df: pd.DataFrame,
        events: List[CompanyEvent],
        window_days: int = 7,
        sector_bias: float = 0.0,
        adjust_position: bool = True,
        adjust_sl_tp: bool = True
    ) -> pd.DataFrame:
        out = df.copy()
        idx = pd.to_datetime(out.index)
        out["Company_Event_Score"] = 0.0
        out["Company_Adjustment"] = 1.0
        out["Company_Expected_Direction"] = None
        out["Company_Confidence"] = 0.0

        for ev in events:
            ev_date = ev.effective_date if ev.effective_date is not None else ev.date
            ev_date = pd.to_datetime(ev_date).normalize()
            start = ev_date
            end = ev_date + pd.Timedelta(days=window_days)
            impact = self.estimate_event_impact(ev, sector_bias=sector_bias)
            mask = (idx >= start) & (idx <= end)
            if not mask.any() and idx.size>0 and ev_date > idx.max():
                mask = (idx == idx.max())
            out.loc[mask, "Company_Event_Score"] = out.loc[mask, "Company_Event_Score"].astype(float) + impact
            out.loc[mask, "Company_Expected_Direction"] = ev.expected_direction
            out.loc[mask, "Company_Confidence"] = np.maximum(out.loc[mask, "Company_Confidence"].astype(float), float(ev.confidence))

        multipliers = 1.0 + out["Company_Event_Score"]
        multipliers = multipliers.clip(lower=self.clamp_min, upper=self.clamp_max).fillna(1.0)
        out["Company_Adjustment"] = multipliers

        if adjust_position:
            if "Position_Size_Pct" in out.columns:
                out["Position_Size_Pct"] = out["Position_Size_Pct"].fillna(0.0) * out["Company_Adjustment"]
            else:
                out["Position_Size_Pct"] = 0.0
                out.loc[out["Company_Adjustment"] > 1.0, "Position_Size_Pct"] = 0.01 * out.loc[out["Company_Adjustment"] > 1.0, "Company_Adjustment"]

        if adjust_sl_tp:
            if "Stop_Loss" in out.columns and "Entry_Price" in out.columns:
                entry = out["Entry_Price"].astype(float)
                sl = out["Stop_Loss"].astype(float)
                tp = out["Take_Profit"].astype(float) if "Take_Profit" in out.columns else None
                # positive news -> expand TP / tighten SL; negative -> tighten TP / widen SL
                score = out["Company_Event_Score"].fillna(0.0)
                pos_factor = 1.0 + score.clip(lower=0.0)
                neg_factor = 1.0 + (-score.clip(upper=0.0))
                long_mask = out["Signal"] == "BUY"
                short_mask = out["Signal"] == "SELL"
                sl_dist = (entry - sl).abs()
                new_sl_dist = sl_dist * neg_factor / pos_factor
                out.loc[long_mask, "Stop_Loss"] = (entry - new_sl_dist).where(long_mask, out.loc[long_mask, "Stop_Loss"])
                out.loc[short_mask, "Stop_Loss"] = (entry + new_sl_dist).where(short_mask, out.loc[short_mask, "Stop_Loss"])
                if tp is not None:
                    tp_dist = (tp - entry).abs()
                    new_tp_dist = tp_dist * pos_factor / neg_factor
                    out.loc[long_mask, "Take_Profit"] = (entry + new_tp_dist).where(long_mask, out.loc[long_mask, "Take_Profit"])
                    out.loc[short_mask, "Take_Profit"] = (entry - new_tp_dist).where(short_mask, out.loc[short_mask, "Take_Profit"])

        out["Company_Adjustment"] = out["Company_Adjustment"].fillna(1.0).clip(self.clamp_min, self.clamp_max)
        out["Company_Event_Score"] = out["Company_Event_Score"].fillna(0.0)
        out["Company_Confidence"] = out["Company_Confidence"].fillna(0.0)
        return out
```# filepath: d:\TradingProjcet\ai-stock-predictor\partC_strategy\company_event_impact.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Iterable
import pandas as pd
import numpy as np

# Optional sentiment helper
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

@dataclass
class CompanyEvent:
    id: str
    date: pd.Timestamp
    headline: str
    event_type: str           # e.g., 'product_launch','bankruptcy','expansion','earnings','guidance','m_and_a','lawsuit','recall','ceo_change'
    expected_direction: str  # 'POSITIVE'|'NEGATIVE'|'MIXED'
    severity: float = 1.0     # 0..3
    confidence: float = 0.5   # 0..1
    effective_date: Optional[pd.Timestamp] = None
    notes: Optional[str] = None

class CompanyEventImpactModel:
    """
    Detect and apply company-specific event impacts:
      - detect events from headlines or load CSV
      - estimate signed impact score
      - apply adjustments to signals DataFrame (Company_Event_Score, Company_Adjustment, Company_Expected_Direction, Company_Confidence)
      - optionally scale Position_Size_Pct and adjust SL/TP
    """
    DEFAULT_SIGNATURES = {
        "product_launch": {"dir": "POSITIVE", "base": 0.5},
        "bankruptcy": {"dir": "NEGATIVE", "base": -2.0},
        "expansion": {"dir": "POSITIVE", "base": 0.6},
        "earnings_beat": {"dir": "POSITIVE", "base": 1.0},
        "earnings_miss": {"dir": "NEGATIVE", "base": -1.0},
        "guidance_up": {"dir": "POSITIVE", "base": 0.8},
        "guidance_down": {"dir": "NEGATIVE", "base": -0.8},
        "m_and_a": {"dir": "MIXED", "base": 0.2},
        "lawsuit": {"dir": "NEGATIVE", "base": -0.7},
        "recall": {"dir": "NEGATIVE", "base": -0.9},
        "ceo_change": {"dir": "MIXED", "base": 0.0},
        "other": {"dir": "MIXED", "base": 0.0}
    }

    def __init__(self, clamp_min: float = 0.05, clamp_max: float = 3.0):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def load_events_from_csv(self, path: str, date_col: str = "date") -> List[CompanyEvent]:
        df = pd.read_csv(path)
        events: List[CompanyEvent] = []
        for i, r in df.iterrows():
            try:
                d = pd.to_datetime(r.get(date_col))
                ev = CompanyEvent(
                    id=str(r.get("id", f"csv_{i}")),
                    date=d,
                    headline=str(r.get("headline", "")),
                    event_type=str(r.get("event_type", "other")),
                    expected_direction=str(r.get("expected_direction", "MIXED")).upper(),
                    severity=float(r.get("severity", 1.0)) if pd.notna(r.get("severity", None)) else 1.0,
                    confidence=float(r.get("confidence", 0.5)) if pd.notna(r.get("confidence", None)) else 0.5,
                    effective_date=pd.to_datetime(r.get("effective_date")) if pd.notna(r.get("effective_date", None)) else None,
                    notes=str(r.get("notes", ""))
                )
                events.append(ev)
            except Exception:
                continue
        return events

    def detect_events_from_headlines(
        self,
        headlines: Iterable[Dict[str, Any]],
        keywords_map: Optional[Dict[str, List[str]]] = None,
        date_key: str = "date",
        text_key: str = "headline"
    ) -> List[CompanyEvent]:
        if keywords_map is None:
            keywords_map = {
                "product_launch": ["launch", "introduce", "releases", "debut", "announces new product"],
                "bankruptcy": ["bankrupt", "bankruptcy", "insolvency", "filed for bankruptcy", "npa"],
                "expansion": ["expansion", "opens", "new plant", "capacity", "acquires facility", "investment"],
                "earnings_beat": ["beats estimates", "beats expectations", "earnings beat", "beat eps", "surpasses estimates"],
                "earnings_miss": ["misses estimates", "missed expectations", "earnings miss", "miss eps"],
                "guidance_up": ["raises guidance", "upgrades guidance", "increases outlook"],
                "guidance_down": ["lowers guidance", "cuts guidance", "reduces outlook"],
                "m_and_a": ["acquire", "merger", "m&a", "takeover", "buyout"],
                "lawsuit": ["sued", "lawsuit", "settlement", "investigation"],
                "recall": ["recall", "withdraw product", "safety issue"],
                "ceo_change": ["ceo", "chief executive", "appoints new", "resigns", "stepping down"]
            }
        events: List[CompanyEvent] = []
        for i, item in enumerate(headlines):
            text = str(item.get(text_key, "")).lower()
            date = pd.to_datetime(item.get(date_key, datetime.utcnow()))
            matched = False
            for etype, kws in keywords_map.items():
                for kw in kws:
                    if kw.lower() in text:
                        sig = self.DEFAULT_SIGNATURES.get(etype, self.DEFAULT_SIGNATURES["other"])
                        expected = sig["dir"]
                        # crude sentiment with VADER if available to refine expected/confidence
                        confidence = 0.6
                        severity = 1.0
                        if _VADER:
                            vs = _VADER.polarity_scores(str(item.get(text_key, "")))
                            # stronger compound -> higher confidence and adjust expected
                            comp = vs.get("compound", 0.0)
                            confidence = min(1.0, 0.4 + abs(comp))
                            if comp > 0.4:
                                expected = "POSITIVE"
                            elif comp < -0.4:
                                expected = "NEGATIVE"
                            severity = 1.0 + min(2.0, abs(comp) * 3.0)
                        ev = CompanyEvent(
                            id=f"det_{i}_{etype}",
                            date=pd.to_datetime(date),
                            headline=item.get(text_key, ""),
                            event_type=etype,
                            expected_direction=expected,
                            severity=severity,
                            confidence=confidence,
                            effective_date=None,
                            notes=f"keyword:{kw}"
                        )
                        events.append(ev)
                        matched = True
                        break
                if matched:
                    break
        return events

    def estimate_event_impact(self, event: CompanyEvent, sector_bias: float = 0.0) -> float:
        sig = self.DEFAULT_SIGNATURES.get(event.event_type, self.DEFAULT_SIGNATURES["other"])
        base = float(sig.get("base", 0.0))
        impact = base * float(event.severity) * float(event.confidence) * (1.0 + float(sector_bias))
        # small floor
        if abs(impact) < 1e-6:
            impact = 0.0
        return float(impact)

    def apply_event_adjustments(
        self,
        df: pd.DataFrame,
        events: List[CompanyEvent],
        window_days: int = 7,
        sector_bias: float = 0.0,
        adjust_position: bool = True,
        adjust_sl_tp: bool = True
    ) -> pd.DataFrame:
        out = df.copy()
        idx = pd.to_datetime(out.index)
        out["Company_Event_Score"] = 0.0
        out["Company_Adjustment"] = 1.0
        out["Company_Expected_Direction"] = None
        out["Company_Confidence"] = 0.0

        for ev in events:
            ev_date = ev.effective_date if ev.effective_date is not None else ev.date
            ev_date = pd.to_datetime(ev_date).normalize()
            start = ev_date
            end = ev_date + pd.Timedelta(days=window_days)
            impact = self.estimate_event_impact(ev, sector_bias=sector_bias)
            mask = (idx >= start) & (idx <= end)
            if not mask.any() and idx.size>0 and ev_date > idx.max():
                mask = (idx == idx.max())
            out.loc[mask, "Company_Event_Score"] = out.loc[mask, "Company_Event_Score"].astype(float) + impact
            out.loc[mask, "Company_Expected_Direction"] = ev.expected_direction
            out.loc[mask, "Company_Confidence"] = np.maximum(out.loc[mask, "Company_Confidence"].astype(float), float(ev.confidence))

        multipliers = 1.0 + out["Company_Event_Score"]
        multipliers = multipliers.clip(lower=self.clamp_min, upper=self.clamp_max).fillna(1.0)
        out["Company_Adjustment"] = multipliers

        if adjust_position:
            if "Position_Size_Pct" in out.columns:
                out["Position_Size_Pct"] = out["Position_Size_Pct"].fillna(0.0) * out["Company_Adjustment"]
            else:
                out["Position_Size_Pct"] = 0.0
                out.loc[out["Company_Adjustment"] > 1.0, "Position_Size_Pct"] = 0.01 * out.loc[out["Company_Adjustment"] > 1.0, "Company_Adjustment"]

        if adjust_sl_tp:
            if "Stop_Loss" in out.columns and "Entry_Price" in out.columns:
                entry = out["Entry_Price"].astype(float)
                sl = out["Stop_Loss"].astype(float)
                tp = out["Take_Profit"].astype(float) if "Take_Profit" in out.columns else None
                # positive news -> expand TP / tighten SL; negative -> tighten TP / widen SL
                score = out["Company_Event_Score"].fillna(0.0)
                pos_factor = 1.0 + score.clip(lower=0.0)
                neg_factor = 1.0 + (-score.clip(upper=0.0))
                long_mask = out["Signal"] == "BUY"
                short_mask = out["Signal"] == "SELL"
                sl_dist = (entry - sl).abs()
                new_sl_dist = sl_dist * neg_factor / pos_factor
                out.loc[long_mask, "Stop_Loss"] = (entry - new_sl_dist).where(long_mask, out.loc[long_mask, "Stop_Loss"])
                out.loc[short_mask, "Stop_Loss"] = (entry + new_sl_dist).where(short_mask, out.loc[short_mask, "Stop_Loss"])
                if tp is not None:
                    tp_dist = (tp - entry).abs()
                    new_tp_dist = tp_dist * pos_factor / neg_factor
                    out.loc[long_mask, "Take_Profit"] = (entry + new_tp_dist).where(long_mask, out.loc[long_mask, "Take_Profit"])
                    out.loc[short_mask, "Take_Profit"] = (entry - new_tp_dist).where(short_mask, out.loc[short_mask, "Take_Profit"])

        out["Company_Adjustment"] = out["Company_Adjustment"].fillna(1.0).clip(self.clamp_min, self.clamp_max)
        out["Company_Event_Score"] = out["Company_Event_Score"].fillna(0.0)
        out["Company_Confidence"] = out["Company_Confidence"].fillna(0.0)
        return out