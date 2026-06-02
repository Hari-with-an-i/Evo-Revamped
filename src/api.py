import sys
import os
import uuid
from typing import Literal

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import run

app = FastAPI(
    title="Evo-Revamped API",
    description="API for fetching narrative analysis sections",
    version="0.1.0"
)

# In-memory store: job_id -> {"status": "...", "result": dict, "error": str}
JOBS = {}

class AnalyzeRequest(BaseModel):
    raw_input: str
    input_type: Literal["claim", "raw_text"] = "claim"

class JobResponse(BaseModel):
    job_id: str
    status: str

def run_analysis_task(job_id: str, raw_input: str, input_type: str):
    """Background task to run the LangGraph pipeline."""
    try:
        result = run(raw_input, input_type)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

@app.post("/api/analyze", response_model=JobResponse)
def start_analysis(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Start a new narrative analysis job. 
    Returns a job_id that can be polled for status and used to fetch results.
    """
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running"}
    background_tasks.add_task(run_analysis_task, job_id, req.raw_input, req.input_type)
    return JobResponse(job_id=job_id, status="running")

@app.get("/api/jobs/{job_id}/status")
def get_status(job_id: str):
    """Get the current status of an analysis job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    return {"job_id": job_id, "status": job["status"], "error": job.get("error")}

def _get_completed_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")
    return job

def _get_narrative_report(job_id: str):
    job = _get_completed_job(job_id)
    report_dict = job["result"].get("narrative_report")
    if not report_dict:
        raise HTTPException(status_code=500, detail="Narrative report not found in job results. The pipeline may have failed or degraded.")
    return report_dict

@app.get("/api/jobs/{job_id}/temporal")
def get_temporal(job_id: str):
    """Section 1 — Temporal Evolution (Buckets, Sentiment, Frames & Voice)"""
    report = _get_narrative_report(job_id)
    return {
        "time_buckets": report.get("time_buckets", []),
        "sentiment_timeline": report.get("sentiment_timeline", []),
        "frame_evolution_log": report.get("frame_evolution_log", []),
        "voice_composition_shifts": report.get("voice_composition_shifts", [])
    }

@app.get("/api/jobs/{job_id}/perspectives")
def get_perspectives(job_id: str):
    """Section 2 — Perspective Landscape & Inflections"""
    report = _get_narrative_report(job_id)
    return {
        "perspective_landscape": report.get("perspective_landscape", []),
        "inflection_point_cards": report.get("inflection_point_cards", [])
    }

@app.get("/api/jobs/{job_id}/analysis")
def get_analysis(job_id: str):
    """Section 3 — Final Analysis [UNRESOLVABLE/VERIFIABLE/CONTESTED]"""
    report = _get_narrative_report(job_id)
    return {
        "claim_snapshot": report.get("claim_snapshot", {}),
        "ground_truth_tier": report.get("ground_truth_tier", "unknown"),
        "narrative_intelligence_summary": report.get("narrative_intelligence_summary", "")
    }

@app.get("/api/jobs/{job_id}/corpus")
def get_corpus(job_id: str):
    """Retrieved Articles Corpus"""
    job = _get_completed_job(job_id)
    articles = job["result"].get("retrieved_articles", [])
    return {"articles": articles}

@app.get("/api/jobs/{job_id}/report")
def get_full_report(job_id: str):
    """Get the full structured Narrative Report as a single JSON object"""
    return _get_narrative_report(job_id)
