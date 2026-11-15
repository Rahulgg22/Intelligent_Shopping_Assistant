from __future__ import annotations

import ipaddress
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or value == "":
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _url_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    parsed = urlparse(str(value).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must use http or https")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".local"):
        raise ValueError("URL host is not allowed")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and (
        address.is_private or address.is_loopback or address.is_link_local or address.is_reserved
    ):
        raise ValueError("URL host is not allowed")
    return str(value).strip()


def _number_value(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return float(value.replace(",", "").strip())
    return value


class Product(BaseModel):
    """Provider-neutral product representation used by retrieval and APIs."""

    model_config = ConfigDict(extra="allow")

    product_id: str = Field(min_length=1)
    name: str = ""
    category: str = ""
    subcategory: str = ""
    brand: str = ""
    price: float | None = Field(default=None, ge=0)
    currency: str | None = "INR"
    original_price: float | None = Field(default=None, ge=0)
    description: str = ""
    features: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    rating: float | None = Field(default=None, ge=0, le=5)
    review_count: int | None = Field(default=None, ge=0)
    in_stock: bool = True
    stock_quantity: int | None = Field(default=None, ge=0)
    discount_percent: float | None = Field(default=None, ge=0, le=100)
    image_url: str | None = None
    product_url: str | None = None
    source_name: str = "local"
    source_product_id: str | None = None
    availability: Literal["in_stock", "out_of_stock", "preorder", "unknown"] = "unknown"
    price_verified: bool = False
    availability_verified: bool = False
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    match_score: float | None = None
    match_reasons: list[str] = Field(default_factory=list)
    model_number: str | None = None
    data_provenance: dict[str, Any] = Field(default_factory=dict)
    weight_kg: float | None = Field(default=None, ge=0)
    color: str = ""
    warranty_months: int | None = Field(default=None, ge=0)
    specifications: str = ""

    @field_validator("features", "use_cases", "tags", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _list_value(value)

    @field_validator("in_stock", mode="before")
    @classmethod
    def normalize_stock(cls, value: Any) -> bool:
        return _bool_value(value)

    @field_validator("image_url", "product_url", mode="before")
    @classmethod
    def normalize_urls(cls, value: Any) -> str | None:
        return _url_value(value)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Product":
        values = dict(record)
        values["price"] = _number_value(values.get("price"))
        values["original_price"] = _number_value(values.get("original_price"))
        values["in_stock"] = _bool_value(values.get("in_stock", True))
        values.setdefault("source_product_id", values.get("product_id"))
        values.setdefault("availability", "in_stock" if values["in_stock"] else "out_of_stock")
        values.setdefault("price_verified", values.get("price") not in (None, ""))
        values.setdefault("availability_verified", "in_stock" in values)
        if "stock_quantity" not in values or values["stock_quantity"] in (None, ""):
            values["stock_quantity"] = 1 if values["in_stock"] else 0
        return cls.model_validate(values)

    def identity_key(self) -> str:
        source_id = self.specifications or self.source_product_id or self.product_id
        return "|".join(part.strip().lower() for part in (self.brand, self.name, source_id))

    def searchable_text(self) -> str:
        return " ".join(
            [
                self.name,
                self.category,
                self.subcategory,
                self.brand,
                self.description,
                *self.features,
                *self.use_cases,
                *self.tags,
            ]
        ).lower()


class ProductSearchResult(BaseModel):
    products: list[Product] = Field(default_factory=list)
    providers: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)
    provider_status: dict[str, str] = Field(default_factory=dict)