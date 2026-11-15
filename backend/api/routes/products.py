from fastapi import APIRouter, HTTPException, Query

from backend.services.product_sources import build_product_service


router = APIRouter(prefix="/api/v1/products", tags=["Products"])
@router.get("")
def list_products(
    q: str = "",
    category: str | None = None,
    subcategory: str | None = None,
    min_price: float | None = Query(default=None, ge=0),
    max_price: float | None = Query(default=None, ge=0),
    in_stock_only: bool = True,
    sort: str = Query(default="relevance", pattern="^(relevance|price_asc|price_desc|rating)$"),
):
    result = build_product_service().search(
        q,
        category=category,
        subcategory=subcategory,
        min_price=min_price,
        max_price=max_price,
        in_stock_only=in_stock_only,
        limit=50,
    )
    products = [product.model_dump(mode="json") for product in result.products]
    if sort == "price_asc":
        products.sort(key=lambda product: product.get("price") is None or product.get("price"))
    elif sort == "price_desc":
        products.sort(key=lambda product: product.get("price") or 0, reverse=True)
    elif sort == "rating":
        products.sort(key=lambda product: product.get("rating") or 0, reverse=True)
    return {
        "products": products,
        "count": len(products),
        "provider_status": result.provider_status,
        "retrieval_errors": result.errors,
    }


@router.get("/{product_id}")
def get_product(product_id: str):
    product = build_product_service().get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found.")
    return product.model_dump(mode="json")