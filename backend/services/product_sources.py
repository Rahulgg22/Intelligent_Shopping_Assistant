from backend.providers.composite import CompositeProductProvider
from backend.providers.serper import SerperShoppingProvider
from backend.providers.web_search import TavilyWebSearchProvider
from backend.services.product_retrieval import ProductRetrievalService


def build_product_service() -> ProductRetrievalService:
    """Build the production service from configured external product sources only."""
    providers = [
        SerperShoppingProvider(),
        TavilyWebSearchProvider(),
    ]
    return ProductRetrievalService(CompositeProductProvider(providers))
