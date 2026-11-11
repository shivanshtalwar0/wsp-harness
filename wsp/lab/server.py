"""HTTP front of the lab: analyst tools under /api, operator tools under /admin, viewer at /.

The MCP server (``wsp.lab.mcp_server``) is a thin proxy onto /api, so the analyst
and the 3D viewer see the same live model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import asyncio
import json

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .session import Lab

VIEWER = Path(__file__).parent / "viewer"

ANALYST_TOOLS = {
    "info": "info",
    "python": "python",
    "practice": "practice",
    "forward_generate": "forward_generate",
    "forward_read": "forward_read",
    "apply_patch": "apply_patch",
    "diff": "diff",
    "checkpoint": "checkpoint",
    "restore": "restore",
    "checkpoints": "checkpoints",
    "note": "note",
    "budget": "get_budget",
    "validation": "validation_for_analyst",
}

ADMIN_TOOLS = {"evaluate": "evaluate", "export": "export", "set_budget": "set_budget",
               "validation": "validation"}


def create_app(lab: Lab) -> FastAPI:
    app = FastAPI(title="wsp lab", version="0.1")

    def dispatch(table: dict, tool: str, body: dict) -> Any:
        if tool not in table:
            raise HTTPException(404, f"unknown tool {tool!r}; known: {sorted(table)}")
        fn = getattr(lab, table[tool])
        try:
            return JSONResponse(_jsonable(fn(**(body or {}))))
        except TypeError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/{tool}")
    def api(tool: str, body: dict = Body(default={})):
        return dispatch(ANALYST_TOOLS, tool, body)

    @app.post("/admin/{tool}")
    def admin(tool: str, body: dict = Body(default={})):
        return dispatch(ADMIN_TOOLS, tool, body)

    @app.get("/viz/layers")
    def viz_layers(mode: str = "weights"):
        return lab.viz_layers(mode)

    @app.get("/viz/neuron")
    def viz_neuron(layer: int, idx: int):
        return _jsonable(lab.viz_neuron(layer, idx))

    @app.get("/viz/state")
    def viz_state(since: int = 0):
        return _jsonable(lab.viz_state(since))

    @app.get("/viz/stream")
    async def viz_stream(request: Request, since: int = 0):
        """Server-sent events: one message whenever lab state changes (checked every 0.2 s),
        plus a 1 s heartbeat while a call is running so the viewer's timer ticks. Reads only
        lock-free state, so it never waits on the model."""
        async def gen():
            n, last = since, None
            while not await request.is_disconnected():
                st = lab.viz_state(n)
                val = st.get("validation") or []
                key = (st["n_events"], st["wver"], st["acts_ver"], len(st["notes"]), len(val),
                       (st["running"] or {}).get("tool"))
                if key != last or st["running"]:
                    yield f"data: {json.dumps(_jsonable(st))}\n\n"
                    n, last = st["n_events"], key
                await asyncio.sleep(1.0 if st["running"] else 0.2)
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/")
    def index():
        return FileResponse(VIEWER / "index.html")

    app.mount("/static", StaticFiles(directory=VIEWER), name="static")
    return app


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    return x
