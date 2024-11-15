from fastapi import APIRouter
from api.api_v0.endpoints import stocks, items

router = APIRouter()

router.include_router(stocks.router)
router.include_router(items.router)