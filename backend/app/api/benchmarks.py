from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.services.benchmark import BenchmarkService

router = APIRouter()

@router.get("/")
def get_benchmarks():
    """获取所有benchmark样例"""
    benchmarks = BenchmarkService.get_benchmarks()
    return benchmarks

@router.get("/{benchmark_id}")
def get_benchmark(benchmark_id: str):
    """获取特定的benchmark样例"""
    benchmark = BenchmarkService.get_benchmark(benchmark_id)
    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return benchmark
