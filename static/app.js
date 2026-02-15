const documentsEl = document.getElementById("documents");
const annotationsEl = document.getElementById("annotations");
const uploadForm = document.getElementById("upload-form");
const commentForm = document.getElementById("comment-form");
const fileInput = document.getElementById("file-input");
const docTitle = document.getElementById("doc-title");
const downloadLink = document.getElementById("download-link");
const pdfContainer = document.getElementById("pdf-container");
const commentsListEl = document.getElementById("annotations");
const commentsSectionEl = document.querySelector("section.comments");
const placementStatusEl = document.getElementById("placement-status");
const commentFilterInput = document.getElementById("comment-filter");
const pageInput = document.getElementById("page");
const xInput = document.getElementById("x");
const yInput = document.getElementById("y");

let selectedDocId = null;
let socket = null;
let pingTimer = null;
let renderToken = 0;
let placement = null;
let activeAnnotationKey = null;
let commentFilterQuery = "";

let currentAnnotations = [];
const annotationItemByKey = new Map();
const pageViews = new Map();

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

function formatDate(dateValue) {
  if (!dateValue) return "";
  return new Date(dateValue).toLocaleString();
}

function annotationKey(annotation, index) {
  if (annotation.id) return annotation.id;
  const rectPart = (annotation.rect || []).map((n) => Number(n).toFixed(2)).join(",");
  return `legacy-${annotation.page}-${rectPart}-${index}`;
}

function setPlacement(page, x, y) {
  placement = { page, x, y };
  pageInput.value = String(page);
  xInput.value = String(x);
  yInput.value = String(y);
  placementStatusEl.textContent = `Selected placement: page ${page}, x ${x.toFixed(1)}, y ${y.toFixed(1)}`;
  renderPlacementPin();
}

function clearPlacement() {
  placement = null;
  pageInput.value = "";
  xInput.value = "";
  yInput.value = "";
  placementStatusEl.textContent = "No placement selected. Click on the PDF.";
  renderPlacementPin();
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail || "Request failed");
  }
  return response.json();
}

function closeSocket() {
  if (pingTimer) {
    clearInterval(pingTimer);
    pingTimer = null;
  }

  if (socket) {
    socket.close();
    socket = null;
  }
}

function connectSocket(docId) {
  closeSocket();
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws/documents/${docId}`);
  socket.onmessage = () => {
    refreshAnnotations();
  };
  socket.onopen = () => {
    socket.send("ping");
    pingTimer = setInterval(() => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send("ping");
      }
    }, 15000);
  };
  socket.onclose = () => {
    if (pingTimer) {
      clearInterval(pingTimer);
      pingTimer = null;
    }
  };
}

function renderDocuments(docs) {
  documentsEl.innerHTML = "";
  for (const doc of docs) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.textContent = `${doc.name} (${formatDate(doc.created_at)})`;
    button.className = doc.id === selectedDocId ? "active" : "";
    button.onclick = async () => {
      selectedDocId = doc.id;
      docTitle.textContent = doc.name;
      downloadLink.href = `/api/documents/${selectedDocId}/file`;
      downloadLink.hidden = false;
      clearPlacement();
      renderDocuments(docs);
      await loadPdf();
      await refreshAnnotations();
      connectSocket(doc.id);
    };
    li.appendChild(button);
    documentsEl.appendChild(li);
  }
}

function flashAnnotationItem(element) {
  element.classList.add("flash");
  window.setTimeout(() => {
    element.classList.remove("flash");
  }, 1400);
}

function scrollItemInCommentsPanel(item) {
  if (!commentsListEl) {
    return;
  }

  const targetContainer =
    commentsListEl.scrollHeight > commentsListEl.clientHeight
      ? commentsListEl
      : commentsSectionEl && commentsSectionEl.scrollHeight > commentsSectionEl.clientHeight
        ? commentsSectionEl
        : commentsListEl;

  const containerRect = targetContainer.getBoundingClientRect();
  const itemRect = item.getBoundingClientRect();
  const deltaTop = itemRect.top - containerRect.top;
  const targetTop = targetContainer.scrollTop + deltaTop - targetContainer.clientHeight / 2 + itemRect.height / 2;
  const nextTop = Math.max(0, targetTop);

  targetContainer.scrollTop = nextTop;
  if (typeof targetContainer.scrollTo === "function") {
    targetContainer.scrollTo({
      top: nextTop,
      behavior: "smooth",
    });
  }
}

function scrollPdfToAnnotationPage(annotation) {
  const pageView = pageViews.get(annotation.page);
  if (!pageView) {
    return;
  }

  const containerRect = pdfContainer.getBoundingClientRect();
  const pageRect = pageView.wrap.getBoundingClientRect();
  const deltaTop = pageRect.top - containerRect.top;
  const targetTop = pdfContainer.scrollTop + deltaTop - 12;
  const nextTop = Math.max(0, targetTop);

  pdfContainer.scrollTop = nextTop;
  if (typeof pdfContainer.scrollTo === "function") {
    pdfContainer.scrollTo({
      top: nextTop,
      behavior: "smooth",
    });
  }
}

function focusAnnotation(annotation, options = {}) {
  const { jumpPdf = false } = options;
  activeAnnotationKey = annotation._key;
  renderAnnotationPins();

  const item = annotationItemByKey.get(annotation._key);
  if (item) {
    scrollItemInCommentsPanel(item);
    flashAnnotationItem(item);
  }

  if (jumpPdf) {
    scrollPdfToAnnotationPage(annotation);
  }
}

function annotationControls(annotation) {
  const controls = document.createElement("div");
  controls.className = "annotation-controls";

  if (!annotation.id) {
    const note = document.createElement("small");
    note.textContent = "External annotation (no /NM ID): read-only in this UI";
    controls.appendChild(note);
    return controls;
  }

  const editButton = document.createElement("button");
  editButton.textContent = "Edit";
  editButton.onclick = async () => {
    const next = prompt("Update comment", annotation.text);
    if (next === null) return;
    await request(`/api/documents/${selectedDocId}/annotations/${annotation.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: next }),
    });
    await refreshAnnotations();
  };

  const toggleResolveButton = document.createElement("button");
  toggleResolveButton.textContent = annotation.resolved ? "Reopen" : "Resolve";
  toggleResolveButton.onclick = async () => {
    await request(`/api/documents/${selectedDocId}/annotations/${annotation.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resolved: !annotation.resolved }),
    });
    await refreshAnnotations();
  };

  controls.appendChild(editButton);
  controls.appendChild(toggleResolveButton);
  return controls;
}

function getFilteredAnnotations() {
  const query = commentFilterQuery.trim().toLowerCase();
  if (!query) {
    return currentAnnotations;
  }

  return currentAnnotations.filter((annotation) => (annotation.text || "").toLowerCase().includes(query));
}

function renderAnnotations(annotations) {
  annotationsEl.innerHTML = "";
  annotationItemByKey.clear();

  for (const annotation of annotations) {
    const li = document.createElement("li");
    li.className = annotation.resolved ? "resolved" : "";
    li.dataset.annotationKey = annotation._key;
    li.innerHTML = `
      <div class="annotation-head">
        <strong>${annotation.author || "Unknown"}</strong>
        <span>Page ${annotation.page}, ID ${annotation.id || "n/a"}</span>
      </div>
      <p>${annotation.text || "(empty)"}</p>
      <small>Modified: ${annotation.modified || "n/a"}</small>
    `;

    li.addEventListener("click", () => {
      focusAnnotation(annotation, { jumpPdf: true });
    });

    li.appendChild(annotationControls(annotation));
    annotationItemByKey.set(annotation._key, li);
    annotationsEl.appendChild(li);
  }
}

function addAnnotationPin(overlay, annotation, pageView) {
  const rect = annotation.rect || [];
  if (rect.length < 4) return;

  const pin = document.createElement("button");
  pin.type = "button";
  pin.className = "annotation-pin";
  if (annotation.resolved) {
    pin.classList.add("resolved");
  }
  if (annotation._key === activeAnnotationKey) {
    pin.classList.add("active");
  }

  pin.title = `${annotation.author || "Unknown"}: ${annotation.text || "(empty)"}`;
  pin.style.left = `${rect[0] * pageView.scale}px`;
  pin.style.top = `${pageView.height - rect[3] * pageView.scale}px`;
  pin.addEventListener("click", (event) => {
    event.stopPropagation();
    focusAnnotation(annotation);
  });

  overlay.appendChild(pin);
}

function renderAnnotationPins() {
  for (const view of pageViews.values()) {
    view.overlay.innerHTML = "";
  }

  for (const annotation of getFilteredAnnotations()) {
    const view = pageViews.get(annotation.page);
    if (!view) continue;
    addAnnotationPin(view.overlay, annotation, view);
  }

  renderPlacementPin();
}

function renderPlacementPin() {
  for (const view of pageViews.values()) {
    const previous = view.overlay.querySelector(".placement-pin");
    if (previous) {
      previous.remove();
    }
  }

  if (!placement) return;
  const view = pageViews.get(placement.page);
  if (!view) return;

  const pin = document.createElement("div");
  pin.className = "placement-pin";
  pin.style.left = `${placement.x * view.scale}px`;
  pin.style.top = `${view.height - placement.y * view.scale}px`;
  view.overlay.appendChild(pin);
}

function onPageClicked(event, pageIndex, pageView) {
  if (event.target !== pageView.canvas && event.target !== pageView.overlay) {
    return;
  }

  const rect = pageView.canvas.getBoundingClientRect();
  const xPx = event.clientX - rect.left;
  const yPx = event.clientY - rect.top;

  if (xPx < 0 || yPx < 0 || xPx > rect.width || yPx > rect.height) {
    return;
  }

  const pdfX = xPx / pageView.scale;
  const pdfY = (pageView.height - yPx) / pageView.scale;
  setPlacement(pageIndex, pdfX, pdfY);
}

async function loadPdf() {
  if (!selectedDocId) {
    pdfContainer.innerHTML = "";
    return;
  }

  const token = ++renderToken;
  pageViews.clear();
  pdfContainer.innerHTML = "Loading PDF...";

  try {
    const url = `/api/documents/${selectedDocId}/file?ts=${Date.now()}`;
    const pdf = await pdfjsLib.getDocument(url).promise;
    if (token !== renderToken) return;

    pdfContainer.innerHTML = "";

    const targetWidth = Math.max(320, pdfContainer.clientWidth - 32);

    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
      const page = await pdf.getPage(pageNumber);
      if (token !== renderToken) return;

      const baseViewport = page.getViewport({ scale: 1 });
      const scale = Math.min(2, Math.max(0.8, targetWidth / baseViewport.width));
      const viewport = page.getViewport({ scale });

      const wrap = document.createElement("div");
      wrap.className = "page-wrap";

      const canvas = document.createElement("canvas");
      canvas.width = Math.floor(viewport.width);
      canvas.height = Math.floor(viewport.height);

      const overlay = document.createElement("div");
      overlay.className = "page-overlay";
      overlay.style.width = `${canvas.width}px`;
      overlay.style.height = `${canvas.height}px`;

      wrap.appendChild(canvas);
      wrap.appendChild(overlay);
      pdfContainer.appendChild(wrap);

      const pageIndex = pageNumber - 1;
      const view = { wrap, canvas, overlay, scale, height: canvas.height };
      pageViews.set(pageIndex, view);

      const context = canvas.getContext("2d");
      await page.render({ canvasContext: context, viewport }).promise;

      wrap.addEventListener("click", (event) => {
        onPageClicked(event, pageIndex, view);
      });
    }

    renderAnnotationPins();
  } catch (error) {
    pdfContainer.textContent = `Failed to render PDF: ${error.message}`;
  }
}

async function refreshAnnotations() {
  if (!selectedDocId) {
    annotationsEl.innerHTML = "";
    currentAnnotations = [];
    return;
  }

  const raw = await request(`/api/documents/${selectedDocId}/annotations`);
  currentAnnotations = raw.map((annotation, index) => ({
    ...annotation,
    _key: annotationKey(annotation, index),
  }));

  renderAnnotations(getFilteredAnnotations());
  renderAnnotationPins();
}

async function loadDocuments() {
  const docs = await request("/api/documents");
  renderDocuments(docs);
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  const response = await fetch("/api/documents", { method: "POST", body: formData });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    alert(payload.detail || "Upload failed");
    return;
  }

  fileInput.value = "";
  await loadDocuments();
});

commentForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedDocId) {
    alert("Select a document first");
    return;
  }

  if (!placement) {
    alert("Click on the PDF to choose where to place this comment.");
    return;
  }

  const payload = {
    author: document.getElementById("author").value,
    page: placement.page,
    x: placement.x,
    y: placement.y,
    text: document.getElementById("text").value,
  };

  await request(`/api/documents/${selectedDocId}/annotations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  document.getElementById("text").value = "";
  clearPlacement();
  await refreshAnnotations();
});

commentFilterInput.addEventListener("input", () => {
  commentFilterQuery = commentFilterInput.value || "";
  renderAnnotations(getFilteredAnnotations());
  renderAnnotationPins();
});

window.addEventListener("resize", () => {
  if (selectedDocId) {
    loadPdf().then(() => {
      renderAnnotationPins();
    });
  }
});

loadDocuments().catch((err) => {
  alert(err.message);
});
