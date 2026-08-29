from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from api_services.inference import (
    IntegrationError,
    run_powerguard_analysis,
)

# 1. استيراد القالب الإحترافي للواجهة من ملف generate_html
from generate_html import rendered_template


# ============================================================
# 1. Environment
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "power_anomaly_model.pkl"
MODEL_PATH = BASE_DIR / MODEL_FILENAME


# ============================================================
# 2. FastAPI Application
# ============================================================

app = FastAPI(
    title="PowerPulse Anomaly Detection API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. Request Model
# ============================================================

class AnalysisRequest(BaseModel):
    state: str = Field(..., min_length=1)
    current_consumption_kwh: float = Field(..., ge=0)
    historical_baseline_kwh: float = Field(..., gt=0)


# ============================================================
# 4. Root (تعديل المسار الرئيسي لعرض واجهة الـ HTML)
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=rendered_template)


# ============================================================
# 5. Health Check
# ============================================================

@app.get("/health")
async def health() -> Dict[str, Any]:
    model_exists = MODEL_PATH.exists()

    if not model_exists:
        return {
            "status": "error",
            "model": MODEL_FILENAME,
            "model_exists": False,
        }

    return {
        "status": "healthy",
        "model": MODEL_FILENAME,
        "model_exists": True,
    }


# ============================================================
# 6. Main Analysis Endpoint
# ============================================================

@app.post("/api/v1/analyze")
async def analyze_power_anomaly(
    request: AnalysisRequest,
) -> Dict[str, Any]:

    try:
        result = run_powerguard_analysis(
            state=request.state,
            current_consumption=request.current_consumption_kwh,
            avg_past_consumption=request.historical_baseline_kwh,
        )

        return result

    except IntegrationError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except TimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Required file not found: {error}",
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing analysis: {error}",
        )


# ============================================================
# 7. Run Backend
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
