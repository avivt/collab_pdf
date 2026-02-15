import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
    TextStringObject,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
DOCS_FILE = DATA_DIR / "documents.json"

PDF_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Collaborative PDF Comments")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnnotationCreate(BaseModel):
    page: int = Field(..., ge=0)
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    text: str = Field(..., min_length=1)
    author: str = Field(default="Anonymous", min_length=1)


class AnnotationUpdate(BaseModel):
    text: str | None = Field(default=None)
    resolved: bool | None = Field(default=None)


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = {}
        self._lock = threading.Lock()

    async def connect(self, doc_id: str, ws: WebSocket) -> None:
        await ws.accept()
        with self._lock:
            if doc_id not in self._connections:
                self._connections[doc_id] = set()
            self._connections[doc_id].add(ws)

    def disconnect(self, doc_id: str, ws: WebSocket) -> None:
        with self._lock:
            if doc_id in self._connections:
                self._connections[doc_id].discard(ws)
                if not self._connections[doc_id]:
                    del self._connections[doc_id]

    async def broadcast(self, doc_id: str, payload: dict[str, Any]) -> None:
        targets = []
        with self._lock:
            targets = list(self._connections.get(doc_id, set()))

        for ws in targets:
            try:
                await ws.send_json(payload)
            except RuntimeError:
                self.disconnect(doc_id, ws)


manager = ConnectionManager()


def _load_documents() -> list[dict[str, Any]]:
    if not DOCS_FILE.exists():
        return []
    with DOCS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_documents(docs: list[dict[str, Any]]) -> None:
    with DOCS_FILE.open("w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)


def _find_document(doc_id: str) -> dict[str, Any]:
    docs = _load_documents()
    for doc in docs:
        if doc["id"] == doc_id:
            return doc
    raise HTTPException(status_code=404, detail="Document not found")


def _format_pdf_date(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("D:%Y%m%d%H%M%SZ")


def _annotation_to_json(annot: DictionaryObject, page_index: int) -> dict[str, Any]:
    rect = annot.get("/Rect") or []
    subtype = str(annot.get("/Subtype", "")).replace("/", "")

    return {
        "id": str(annot.get("/NM", "")),
        "subtype": subtype,
        "page": page_index,
        "rect": [float(value) for value in rect],
        "text": str(annot.get("/Contents", "")),
        "author": str(annot.get("/T", "")),
        "modified": str(annot.get("/M", "")),
        "resolved": str(annot.get("/State", "")) == "/Accepted",
    }


def _extract_annotations(pdf_path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    merged: dict[str, dict[str, Any]] = {}

    supported_subtypes = {
        "/Text",
        "/FreeText",
        "/Highlight",
        "/Underline",
        "/StrikeOut",
        "/Squiggly",
        "/Caret",
        "/Stamp",
    }

    def _record_key(annotation_json: dict[str, Any]) -> str:
        if annotation_json.get("id"):
            return str(annotation_json["id"])
        rect_part = ",".join(str(value) for value in annotation_json.get("rect", []))
        return f'{annotation_json.get("page")}:{annotation_json.get("subtype")}:{rect_part}'

    def _upsert(annotation_json: dict[str, Any]) -> None:
        key = _record_key(annotation_json)
        existing = merged.get(key)
        if existing is None:
            merged[key] = annotation_json
            return

        if not existing.get("text") and annotation_json.get("text"):
            existing["text"] = annotation_json["text"]
        if not existing.get("author") and annotation_json.get("author"):
            existing["author"] = annotation_json["author"]
        if not existing.get("modified") and annotation_json.get("modified"):
            existing["modified"] = annotation_json["modified"]
        existing["resolved"] = bool(existing.get("resolved") or annotation_json.get("resolved"))

    for page_index, page in enumerate(reader.pages):
        annots = page.get("/Annots")
        if not annots:
            continue
        for annot_ref in annots:
            annot = annot_ref.get_object()
            subtype = str(annot.get("/Subtype", ""))
            if subtype in supported_subtypes:
                _upsert(_annotation_to_json(annot, page_index))
                continue

            # Some editors store insertion comment text on a Popup annotation
            # linked to a parent /Caret (or other markup) annotation.
            if subtype == "/Popup":
                parent_ref = annot.get("/Parent")
                if not parent_ref:
                    continue
                parent = parent_ref.get_object()
                parent_subtype = str(parent.get("/Subtype", ""))
                if parent_subtype not in supported_subtypes:
                    continue

                data = _annotation_to_json(parent, page_index)
                popup_text = str(annot.get("/Contents", ""))
                popup_author = str(annot.get("/T", ""))
                popup_modified = str(annot.get("/M", ""))

                if not data.get("text") and popup_text:
                    data["text"] = popup_text
                if not data.get("author") and popup_author:
                    data["author"] = popup_author
                if not data.get("modified") and popup_modified:
                    data["modified"] = popup_modified
                _upsert(data)

    return list(merged.values())


def _write_pdf_atomic(writer: PdfWriter, destination: Path) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        writer.write(f)
    os.replace(tmp_path, destination)


def _create_text_annotation(payload: AnnotationCreate) -> DictionaryObject:
    now = _format_pdf_date()
    annotation_id = str(uuid.uuid4())
    # A standard sticky-note text annotation for Acrobat/Preview interoperability.
    annotation = DictionaryObject()
    annotation.update(
        {
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Text"),
            NameObject("/Rect"): ArrayObject(
                [
                    FloatObject(payload.x),
                    FloatObject(payload.y),
                    FloatObject(payload.x + 24),
                    FloatObject(payload.y + 24),
                ]
            ),
            NameObject("/Contents"): TextStringObject(payload.text),
            NameObject("/T"): TextStringObject(payload.author),
            NameObject("/M"): TextStringObject(now),
            NameObject("/CreationDate"): TextStringObject(now),
            NameObject("/NM"): TextStringObject(annotation_id),
            NameObject("/Name"): NameObject("/Comment"),
            NameObject("/Open"): BooleanObject(False),
            NameObject("/F"): NumberObject(4),
            NameObject("/C"): ArrayObject([FloatObject(1), FloatObject(1), FloatObject(0)]),
        }
    )
    return annotation


def _add_annotation(pdf_path: Path, payload: AnnotationCreate) -> dict[str, Any]:
    reader = PdfReader(str(pdf_path))
    if payload.page >= len(reader.pages):
        raise HTTPException(status_code=400, detail="Page index out of range")

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    page = writer.pages[payload.page]
    annots = page.get("/Annots")
    if annots is None:
        annots = ArrayObject()
        page[NameObject("/Annots")] = annots

    annotation = _create_text_annotation(payload)
    annot_ref = writer._add_object(annotation)
    annots.append(annot_ref)

    _write_pdf_atomic(writer, pdf_path)
    return _annotation_to_json(annotation, payload.page)


def _update_annotation(pdf_path: Path, annotation_id: str, payload: AnnotationUpdate) -> dict[str, Any]:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    for page_index, page in enumerate(writer.pages):
        annots = page.get("/Annots")
        if not annots:
            continue

        for annot_ref in annots:
            annot = annot_ref.get_object()
            if str(annot.get("/NM", "")) != annotation_id:
                continue

            if payload.text is not None:
                annot[NameObject("/Contents")] = TextStringObject(payload.text)
            annot[NameObject("/M")] = TextStringObject(_format_pdf_date())

            if payload.resolved is not None:
                annot[NameObject("/StateModel")] = NameObject("/Review")
                annot[NameObject("/State")] = NameObject("/Accepted" if payload.resolved else "/None")

            _write_pdf_atomic(writer, pdf_path)
            return _annotation_to_json(annot, page_index)

    raise HTTPException(status_code=404, detail="Annotation not found")


doc_locks: dict[str, threading.Lock] = {}
doc_locks_guard = threading.Lock()


def _get_doc_lock(doc_id: str) -> threading.Lock:
    with doc_locks_guard:
        if doc_id not in doc_locks:
            doc_locks[doc_id] = threading.Lock()
        return doc_locks[doc_id]


@app.get("/")
def root() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/documents")
def list_documents() -> list[dict[str, Any]]:
    return _load_documents()


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    doc_id = str(uuid.uuid4())
    filename = f"{doc_id}.pdf"
    path = PDF_DIR / filename

    content = await file.read()
    with path.open("wb") as f:
        f.write(content)

    # Validate it can be parsed as PDF before accepting.
    try:
        PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid PDF") from exc

    doc = {
        "id": doc_id,
        "name": file.filename,
        "path": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    docs = _load_documents()
    docs.append(doc)
    _save_documents(docs)
    return doc


@app.get("/api/documents/{doc_id}/file")
def get_document_file(doc_id: str) -> FileResponse:
    doc = _find_document(doc_id)
    pdf_path = PDF_DIR / doc["path"]
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="File missing")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=doc["name"],
        content_disposition_type="inline",
    )


@app.get("/api/documents/{doc_id}/annotations")
def list_annotations(doc_id: str) -> list[dict[str, Any]]:
    doc = _find_document(doc_id)
    pdf_path = PDF_DIR / doc["path"]
    return _extract_annotations(pdf_path)


@app.post("/api/documents/{doc_id}/annotations")
async def create_annotation(doc_id: str, payload: AnnotationCreate) -> dict[str, Any]:
    doc = _find_document(doc_id)
    pdf_path = PDF_DIR / doc["path"]

    lock = _get_doc_lock(doc_id)
    with lock:
        created = _add_annotation(pdf_path, payload)

    await manager.broadcast(doc_id, {"type": "annotation_created", "annotation": created})
    return created


@app.patch("/api/documents/{doc_id}/annotations/{annotation_id}")
async def update_annotation(doc_id: str, annotation_id: str, payload: AnnotationUpdate) -> dict[str, Any]:
    doc = _find_document(doc_id)
    pdf_path = PDF_DIR / doc["path"]

    lock = _get_doc_lock(doc_id)
    with lock:
        updated = _update_annotation(pdf_path, annotation_id, payload)

    await manager.broadcast(doc_id, {"type": "annotation_updated", "annotation": updated})
    return updated


@app.websocket("/ws/documents/{doc_id}")
async def ws_document(doc_id: str, websocket: WebSocket) -> None:
    await manager.connect(doc_id, websocket)
    try:
        while True:
            # Keep alive. UI can send lightweight pings.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(doc_id, websocket)


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
