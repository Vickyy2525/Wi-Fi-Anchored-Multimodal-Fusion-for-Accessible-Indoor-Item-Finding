"""Pydantic models for realtime ingest API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    device_id: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str


class WifiRow(BaseModel):
    timestamp: int
    ssid: str = ""
    bssid: str = ""
    level: int = 0
    frequency: int = 0


class WifiBatch(BaseModel):
    rows: list[WifiRow] = Field(default_factory=list)


class ImuRow(BaseModel):
    timestamp: int
    acc_x: Optional[float] = None
    acc_y: Optional[float] = None
    acc_z: Optional[float] = None
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None
    mag_x: Optional[float] = None
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None
    jerk: float = 0.0
    gyro_mag: float = 0.0


class ImuBatch(BaseModel):
    rows: list[ImuRow] = Field(default_factory=list)


class LabelRow(BaseModel):
    timestamp: int
    label: str


class LabelBatch(BaseModel):
    rows: list[LabelRow] = Field(default_factory=list)


class RunWindowRequest(BaseModel):
    session_id: str
    window_id: Optional[int] = None  # null = latest complete window before now


class HealthResponse(BaseModel):
    status: str = "ok"
    graph_backend: str = "file"
    detail: Optional[str] = None


class ObjectQueryResponse(BaseModel):
    result: Any
