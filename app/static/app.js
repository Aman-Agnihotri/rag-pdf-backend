const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const questionInput = document.getElementById("question");
const statusEl = document.getElementById("status");
const uploadInput = document.getElementById("pdf-input");
const uploadBtn = document.getElementById("upload-btn");
const uploadMeta = document.getElementById("upload-meta");

const setStatus = (text) => {
  statusEl.textContent = text;
};

const addMessage = (text, role, meta = "") => {
  const bubble = document.createElement("div");
  bubble.className = `message ${role}`;
  bubble.textContent = text;
  if (meta) {
    const small = document.createElement("small");
    small.textContent = meta;
    bubble.appendChild(small);
  }
  chat.appendChild(bubble);
  chat.scrollTop = chat.scrollHeight;
};

uploadInput.addEventListener("change", () => {
  const files = uploadInput.files;
  if (!files || files.length === 0) {
    uploadMeta.textContent = "No files selected.";
    return;
  }
  uploadMeta.textContent = `${files.length} file(s) ready.`;
});

uploadBtn.addEventListener("click", async () => {
  if (!uploadInput.files || uploadInput.files.length === 0) {
    uploadMeta.textContent = "Please choose PDFs first.";
    return;
  }
  const formData = new FormData();
  Array.from(uploadInput.files).forEach((file) => formData.append("files", file));
  setStatus("Ingesting...");
  try {
    const res = await fetch("/ingest", { method: "POST", body: formData });
    const data = await res.json();
    uploadMeta.textContent = `Ingested ${data.documents_ingested} docs, ${data.chunks_added} chunks.`;
    setStatus("Idle");
  } catch (error) {
    uploadMeta.textContent = "Ingestion failed.";
    setStatus("Error");
  }
});

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  addMessage(question, "user");
  questionInput.value = "";
  setStatus("Thinking...");
  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    const meta = data.citations && data.citations.length
      ? `Sources: ${data.citations.map((c) => `${c.source} p.${c.pages}`).join(" | ")}`
      : "No citations.";
    addMessage(data.answer, "bot", meta);
    setStatus("Idle");
  } catch (error) {
    addMessage("Sorry, something went wrong.", "bot");
    setStatus("Error");
  }
});