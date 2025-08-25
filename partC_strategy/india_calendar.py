from typing import List, Optional
import os
import requests
import json
import pandas as pd
from datetime import datetime
from dataclasses import asdict

# reuse PolicyEvent dataclass from policy_impact
try:
    from partC_strategy.policy_impact import PolicyEvent
except Exception:
    PolicyEvent = None  # will raise below if used

class IndiaPolicyCalendarFetcher:
    """
    Fetch Indian government / central bank policy event calendars and convert to PolicyEvent objects.

    Strategy (in order):
      1) TradingEconomics API if TRADINGECONOMICS_KEY env var set (recommended)
      2) Local CSV fallback at data/india_policy_calendar.csv (columns: date, headline, policy_type, expected_direction, severity, confidence, effective_date)
      3) Returns empty list and logs hint.

    Notes:
      - TradingEconomics endpoint and parameters may require an account/key. Set env TRADINGECONOMICS_KEY.
      - This module does not perform brittle HTML scraping of RBI pages; if you need that, please provide the RBI page URL and I can add a parser.
    """

    def __init__(self, te_key_env: str = "TRADINGECONOMICS_KEY"):
        self.te_key = os.environ.get(te_key_env)

    def fetch_from_tradingeconomics(self, start: Optional[str] = None, end: Optional[str] = None) -> List[PolicyEvent]:
        """
        Query TradingEconomics calendar for India. Returns list of PolicyEvent.
        Requires TRADINGECONOMICS_KEY in environment.
        """
        if self.te_key is None:
            return []

        # build URL (TradingEconomics calendar endpoints vary; this uses a commonly supported path)
        url = "https://api.tradingeconomics.com/calendar/country/india"
        params = {"c": self.te_key}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            items = r.json()
        except Exception:
            return []

        events: List[PolicyEvent] = []
        for it in items:
            # best-effort mapping from TE fields to PolicyEvent
            # TE items typically contain: date, country, category, event, impact, previous, actual
            try:
                date = pd.to_datetime(it.get("date") or it.get("eventDate") or it.get("time"))
            except Exception:
                continue
            headline = it.get("event") or it.get("title") or it.get("description") or ""
            category = (it.get("category") or "").lower()
            # heuristic policy_type mapping
            event_lower = headline.lower()
            if "interest rate" in event_lower or "policy rate" in event_lower or "mpc" in event_lower or "repo rate" in event_lower:
                policy_type = "interest_rate_hike" if (str(it.get("actual", "")).strip() and float(it.get("actual", 0)) >= float(it.get("previous", 0))) else "interest_rate_cut"
                # if no numeric info, mark generic interest_rate
                if str(it.get("actual", "")).strip() == "":
                    policy_type = "interest_rate_hike" if "hike" in event_lower else ("interest_rate_cut" if "cut" in event_lower else "interest_rate")
            elif "budget" in event_lower or "union budget" in event_lower:
                policy_type = "fiscal_stimulus"
            elif "tax" in event_lower:
                policy_type = "tax_increase" if "increase" in event_lower or "raise" in event_lower else "corporate_tax_cut" if "cut" in event_lower else "other"
            else:
                policy_type = "other"

            # estimate expected_direction using actual vs previous (when numeric)
            expected = "MIXED"
            try:
                actual = it.get("actual")
                prev = it.get("previous")
                if actual is not None and prev is not None:
                    a = float(actual)
                    p = float(prev)
                    if policy_type.startswith("interest_rate"):
                        expected = "POSITIVE" if a < p else "NEGATIVE" if a > p else "MIXED"
                    else:
                        # for budget/tax events use simple heuristics
                        expected = "POSITIVE" if "stimulus" in event_lower or "cut" in event_lower else ("NEGATIVE" if "tax" in event_lower and ("increase" in event_lower or "hike" in event_lower) else "MIXED")
            except Exception:
                expected = "MIXED"

            severity = 1.0
            confidence = 0.6 if it.get("impact","").lower() in ("high","h") else 0.4

            ev = PolicyEvent(
                id=f"te_{int(pd.Timestamp(date).timestamp())}_{policy_type}",
                date=pd.to_datetime(date),
                headline=headline,
                policy_type=policy_type,
                expected_direction=expected,
                severity=severity,
                confidence=confidence,
                effective_date=None,
                notes=json.dumps({k: it.get(k) for k in ("actual","previous","impact","category") if k in it})
            )
            events.append(ev)
        return events

    def fetch_from_local_csv(self, path: str = "data/india_policy_calendar.csv") -> List[PolicyEvent]:
        """
        Load a CSV in data/india_policy_calendar.csv if present.
        Expected columns: date, headline, policy_type, expected_direction, severity, confidence, effective_date
        """
        if not os.path.exists(path):
            return []
        try:
            df = pd.read_csv(path)
        except Exception:
            return []
        events: List[PolicyEvent] = []
        for _, r in df.iterrows():
            try:
                date = pd.to_datetime(r.get("date"))
                ev = PolicyEvent(
                    id=str(r.get("id", f"csv_{_}")),
                    date=date,
                    headline=str(r.get("headline", "")),
                    policy_type=str(r.get("policy_type", "other")),
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

    def get_policy_events(self, start: Optional[str] = None, end: Optional[str] = None) -> List[PolicyEvent]:
        """
        Try sources in order and return list of PolicyEvent.
        """
        if PolicyEvent is None:
            raise RuntimeError("PolicyEvent dataclass not available (policy_impact module missing)")

        # 1) TradingEconomics
        te_events = []
        try:
            te_events = self.fetch_from_tradingeconomics(start=start, end=end)
        except Exception:
            te_events = []
        if te_events:
            return te_events

        # 2) Local CSV
        csv_events = self.fetch_from_local_csv()
        if csv_events:
            return csv_events

        # 3) nothing found
        return []

# Example usage:
# from partC_strategy.india_calendar import IndiaPolicyCalendarFetcher
# f = IndiaPolicyCalendarFetcher()
# events = f.get_policy_events()
# Then feed events into PolicyImpactModel.apply_policy_adjustments(...)