
document.addEventListener('DOMContentLoaded', function () {


    document.getElementById('fileInput').addEventListener('change', function (event) {
        const files = event.target.files;
        const container = document.getElementById('imgVisual');
        container.innerHTML = ''; // Clear previous previews

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const reader = new FileReader();

            reader.onload = function (e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'preview-image';
                container.appendChild(img);
            };

            reader.readAsDataURL(file);
        }
    });
});