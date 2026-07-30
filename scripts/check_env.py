"""Проверка окружения. Запуск: python scripts/check_env.py"""
from importlib.metadata import version, PackageNotFoundError

PKGS = [
    "numpy", "scipy", "pandas", "scikit-learn", "matplotlib",
    "lightgbm", "xgboost", "catboost", "shap",
    "xarray", "rioxarray", "rasterio", "geopandas", "shapely",
    "dask", "zarr", "pystac-client", "odc-stac", "verde", "esda", "harmonica",
    "duckdb", "h3",
    "statsforecast", "mlforecast", "neuralforecast", "mapie",
    "simpeg", "discretize", "geoana", "empymod", "geoh5py",
]

def versions():
    print("=== Версии ===")
    missing = []
    for p in PKGS:
        try:
            print(f"{p:20s} {version(p)}")
        except PackageNotFoundError:
            print(f"{p:20s} НЕ УСТАНОВЛЕН")
            missing.append(p)
    return missing

def check_openmp():
    """LightGBM и XGBoost — первое, что падает на маке без libomp."""
    print("\n=== OpenMP ===")
    for mod in ("lightgbm", "xgboost"):
        try:
            __import__(mod)
            print(f"{mod:20s} импортируется")
        except Exception as e:
            print(f"{mod:20s} ОШИБКА: {type(e).__name__}: {e}")
            print("  → brew install libomp")

def check_torch():
    print("\n=== Ускоритель ===")
    try:
        import torch
        print(f"torch {torch.__version__}")
        print(f"MPS доступен:  {torch.backends.mps.is_available()}")
        print(f"MPS собран:    {torch.backends.mps.is_built()}")
        print(f"CUDA доступна: {torch.cuda.is_available()}  (на Apple Silicon всегда False)")
    except ImportError:
        print("torch не установлен — нормально, если группа dl отложена")

def check_solver():
    """На Apple Silicon Pardiso недоступен. Убеждаемся, что есть чем его заменить."""
    print("\n=== Разреженный солвер ===")
    import numpy as np
    from scipy.sparse import diags
    from scipy.sparse.linalg import splu
    A = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(500, 500), format="csc")
    x = splu(A).solve(np.ones(500))
    print(f"SciPy SuperLU:  работает (норма решения {np.linalg.norm(x):.3e})")
    try:
        import pymatsolver
        print(f"pymatsolver:    установлен — {getattr(pymatsolver, 'AvailableSolvers', 'см. dir(pymatsolver)')}")
    except ImportError:
        print("pymatsolver:    не установлен — так и задумано, SimPEG возьмёт SciPy")

def check_cog():
    """Главная проверка дня: GDAL + сеть + HTTP range requests разом."""
    print("\n=== Чтение COG по сети ===")
    import rioxarray  # noqa: F401
    import xarray as xr
    from pystac_client import Client

    cat = Client.open("https://earth-search.aws.element84.com/v1")
    items = list(cat.search(
        collections=["sentinel-2-l2a"],
        bbox=[104.2, 52.2, 104.4, 52.4],   # окрестности Иркутска
        datetime="2025-07-01/2025-08-31",
        query={"eo:cloud_cover": {"lt": 20}},
        max_items=1,
    ).items())
    if not items:
        print("STAC не вернул сцен — расширьте окно дат или облачность")
        return
    href = items[0].assets["red"].href
    print(f"сцена: {items[0].id}")
    da = xr.open_dataarray(href, engine="rasterio").isel(band=0)
    sub = da[:512, :512]
    sub = sub.compute() if hasattr(sub.data, "compute") else sub
    print(f"считан фрагмент {sub.shape}, среднее {float(sub.mean()):.1f}, CRS {da.rio.crs}")

if __name__ == "__main__":
    missing = versions()
    check_openmp()
    check_torch()
    check_solver()
    check_cog()
    print("\nОтсутствуют:", missing or "нет")
