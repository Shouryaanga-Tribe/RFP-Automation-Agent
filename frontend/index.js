const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("rfpUpload");

// Click opens file selector
uploadArea.addEventListener("click", () => fileInput.click());

// Drag styling
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");

  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    uploadArea.textContent = `Uploaded: ${e.dataTransfer.files[0].name}`;
    extractQA(); // Auto-extract on drop
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    uploadArea.textContent = `Uploaded: ${fileInput.files[0].name}`;
    extractQA(); // Auto-extract on select
  }
});

function extractQA() {
  // Placeholder logic for demo
  const dummyQA = `Q: What is the submission deadline?\nA: The submission deadline is July 15, 2025.\n\nQ: Who is eligible?\nA: Only registered vendors.`;
  document.getElementById("qaBox").value = dummyQA;
  document.getElementById("editedQA").value = dummyQA;
}

function generateResponse() {
  const modifiedQA = document.getElementById("editedQA").value;
  const generatedResponse = `Thank you for your interest. Based on your questions:\n\n${modifiedQA}\n\nPlease reach out for any further clarification.`;
  document.getElementById("responseBox").value = generatedResponse;
}

function downloadResponse() {
  const responseText = document.getElementById("responseBox").value;
  const blob = new Blob([responseText], { type: "text/plain" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "RFP_Response.txt";
  link.click();
}

