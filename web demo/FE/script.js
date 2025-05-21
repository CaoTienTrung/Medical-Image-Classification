document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const loadingIndicator = document.getElementById("loadingIndicator");
  const predictionResult = document.getElementById("predictionResult");

  // Handle click on drop zone
  dropZone.addEventListener("click", () => {
    fileInput.click();
  });

  // Handle file selection
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFile(file);
    }
  });

  // Handle drag and drop events
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, highlight, false);
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, unhighlight, false);
  });

  function highlight(e) {
    dropZone.classList.add("dragover");
  }

  function unhighlight(e) {
    dropZone.classList.remove("dragover");
  }

  dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFile(file);
    }
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file");
      return;
    }

    // Display image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      previewContainer.style.display = "grid";
      getPrediction(file); // Gọi hàm mới để gửi yêu cầu thực tế
    };
    reader.readAsDataURL(file);
  }

  async function getPrediction(file) {
    loadingIndicator.style.display = "flex";
    predictionResult.innerHTML = "";

    try {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      const base64String = await new Promise((resolve) => {
        reader.onloadend = () => resolve(reader.result.split(",")[1]);
      });

      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64String }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      const predictions = data.predictions;

      // Hiển thị kết quả
      const resultHTML = predictions
        .map(
          (pred) => `
            <div class="prediction-item">
              <span class="label">${pred.label}</span>
              <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${
                  pred.confidence * 100
                }%"></div>
              </div>
              <span class="confidence-value">${(pred.confidence * 100).toFixed(
                1
              )}%</span>
            </div>
          `
        )
        .join("");

      predictionResult.innerHTML = resultHTML;
    } catch (error) {
      console.error("Error:", error);
      predictionResult.innerHTML =
        "<p>Error getting prediction. Please try again.</p>";
    } finally {
      loadingIndicator.style.display = "none";
    }
  }
});
