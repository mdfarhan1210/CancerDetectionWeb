document.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const captureButton = document.getElementById("capture");
    const imageDataInput = document.getElementById("image-data");
    const fileInput = document.getElementById("file");
    const selectFileButton = document.getElementById("select-file");
    const previewImage = document.getElementById("preview-image");

    // Access the user's webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => console.error("Error accessing webcam:", err));

    // Capture image from the camera
    captureButton.addEventListener("click", () => {
        const context = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert the captured image to base64
        const imageData = canvas.toDataURL("image/jpeg");
        imageDataInput.value = imageData; // Save the image data in the hidden input

        // Show the captured image in the preview area
        previewImage.src = imageData;
        previewImage.style.display = "block";

        alert("Image captured and displayed!");
    });

    // File input trigger
    selectFileButton.addEventListener("click", () => {
        fileInput.click();
    });

    // Show preview for selected files
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                previewImage.src = event.target.result;
                previewImage.style.display = "block";
            };
            reader.readAsDataURL(file);
        }
    });

    // Add drag-and-drop support
    const dropArea = document.getElementById("drop-area");

    dropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropArea.style.borderColor = "blue";
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.style.borderColor = "#ccc";
    });

    dropArea.addEventListener("drop", (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;

            // Display the dragged image
            const reader = new FileReader();
            reader.onload = (event) => {
                previewImage.src = event.target.result;
                previewImage.style.display = "block";
            };
            reader.readAsDataURL(files[0]);

            alert("File added via drag-and-drop!");
        }
    });
});
