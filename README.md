# Collaborative PDF Comments (Native PDF Annotations)

This app lets a team upload PDFs, add/edit text comments, and store comments as standard PDF annotations (`/Annot`, subtype `/Text`) so they remain interoperable with Acrobat and macOS Preview.

## What this MVP supports

- Upload standard PDF files.
- Read existing native annotations from PDFs.
- Add new sticky-note text comments as native PDF annotations.
- Place new comments by clicking directly on the rendered PDF page.
- Edit comment text.
- Resolve/reopen comments using PDF review state (`/StateModel /Review`, `/State /Accepted`).
- Realtime collaboration updates in browser via WebSockets.
- Click annotation pins on the PDF to jump to their entry in the comment list.
- Filter comments by text content using a contains search box.

## Interoperability details

Annotations are written directly into the PDF file using standard fields including:

- `/Type /Annot`
- `/Subtype /Text`
- `/Contents`
- `/T`
- `/M`
- `/CreationDate`
- `/Rect`
- `/NM`

This is the key requirement for Acrobat/Preview interoperability.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

Note: The in-app renderer uses `pdf.js` from a CDN at runtime.

## Notes for production hardening

- Add auth + per-document ACLs.
- Move metadata storage from JSON file to Postgres.
- Add versioning/backup for PDF files.
- Add signed URLs/object storage.
- Add conflict-safe revision IDs and audit trail.
