# components/last_update.py
from __future__ import annotations
from datetime import datetime, date
from typing import Optional, Union

import streamlit as st

Dateish = Union[str, datetime, date, None]

def _fmt_date(d: Dateish) -> str:
    if d is None:
        return ""
    if isinstance(d, str):
        return d
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, date):
        return d.strftime("%Y-%m-%d")
    return str(d)

def show_last_update_badge(
    data_upto: Dateish = None,
    model_version: Optional[str] = None,
    last_train: Dateish = None,
) -> None:
    """
    Başlığın altında tazelik çubuğu gösterir.
    Örnek çıktı: "Veri: 2025-10-05’e kadar • Model v0.3.1 • Son eğitim: 2025-10-04"
    """
    parts = []
    if data_upto:
        parts.append(f"Veri: {_fmt_date(data_upto)}’e kadar")
    if model_version:
        parts.append(f"Model {model_version}")
    if last_train:
        parts.append(f"Son eğitim: {_fmt_date(last_train)}")

    if not parts:
        return

    st.markdown(
        f"""
        <div style="
            background:#0e1117;
            color:#c7d1d8;
            border:1px solid #2b313e;
            padding:8px 12px;
            border-radius:10px;
            display:inline-block;
            font-size:0.95rem;
        ">{' • '.join(parts)}</div>
        """,
        unsafe_allow_html=True,
    )
