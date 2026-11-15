"""Backward-compatible alias for the external product-source builder."""

from backend.services.product_sources import build_product_service


def build_catalog_service():
    return build_product_service()
