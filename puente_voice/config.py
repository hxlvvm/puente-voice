"""Clinic configuration: time zone and visit types with their spoken labels and synonyms."""
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, Field, field_validator


class VisitType(BaseModel):
    code: str
    es: str
    en: str
    synonyms_es: list[str] = []
    synonyms_en: list[str] = []


class ClinicConfig(BaseModel):
    clinic_id: str
    name: str
    timezone: str = "America/Chicago"
    visit_types: list[VisitType]
    max_slots_offered: int = Field(default=3, ge=1, le=5)
    search_days: int = Field(default=14, ge=1, le=90)

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, v: str) -> str:
        ZoneInfo(v)
        return v

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def visit(self, code: str) -> VisitType:
        return next(v for v in self.visit_types if v.code == code)


def load_clinic(path) -> ClinicConfig:
    return ClinicConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
