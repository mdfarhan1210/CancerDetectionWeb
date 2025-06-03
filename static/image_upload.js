document.addEventListener("DOMContentLoaded", () => {
    const dropArea = document.getElementById("drop-area");
    const fileInput = document.getElementById("file");
    
    dropArea.addEventListener("click", () => {
        fileInput.click();
    });

    dropArea.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropArea.classList.add("drag-over");
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.classList.remove("drag-over");
    });

    dropArea.addEventListener("drop", (event) => {
        event.preventDefault();
        dropArea.classList.remove("drag-over");
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            dropArea.querySelector("p").textContent = files[0].name;
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            dropArea.querySelector("p").textContent = fileInput.files[0].name;
        }
    });
});
