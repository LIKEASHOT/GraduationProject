#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASGI entrypoint for the unified single-port backend.
It serves the legacy Flask HTTP APIs and the realtime websocket service
through one FastAPI application.
"""

from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.wsgi import WSGIMiddleware

from flask_app import app as flask_app
from flask_app import init_system
from full_duplex_backends import RealtimeBackendBundle
from full_duplex_session import FullDuplexSession


app = FastAPI(title="EchoSage Unified Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

realtime_backends = RealtimeBackendBundle()


@app.on_event("startup")
async def startup_event() -> None:
    init_system()
    realtime_backends.warmup()


@app.get("/health")
async def health() -> dict:
    return {
        "success": True,
        "message": "EchoSage realtime backend is running",
        "preserves_text_chat": True,
    }


@app.websocket("/ws/full-duplex")
async def full_duplex_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    session = FullDuplexSession(websocket, realtime_backends)
    try:
        await session.run()
    except WebSocketDisconnect:
        return


app.mount("/", WSGIMiddleware(flask_app))
