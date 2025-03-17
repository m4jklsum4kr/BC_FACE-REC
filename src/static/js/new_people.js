
document.addEventListener('DOMContentLoaded', function() {
    initializeStepClickListener();
    setCurrentStep(1);
    initializeFileUploadListener();
});

function setCurrentStep(stepNumber) {
    const steps = document.querySelectorAll('.arrow-steps .step');
    const panels = document.querySelectorAll('.panel');
    steps.forEach((step, index) => {
        if (index === stepNumber - 1) {
            step.classList.add('current');
        } else {
            step.classList.remove('current');
        }
    });
    panels.forEach((panel, index) => {
        if (index === stepNumber - 1) {
            panel.classList.remove('hidden');
        } else {
            panel.classList.add('hidden');
        }
    });
}

function initializeStepClickListener() {
    const steps = document.querySelectorAll('.arrow-steps .step');
    steps.forEach((step, index) => {
        step.addEventListener('click', function() {
            setCurrentStep(index + 1);
        });
    });
}



function initializeFileUploadListener() {
    document.getElementById('fileInput').addEventListener('change', function (event) {
        const files = event.target.files;
        const container = document.getElementById('upload-container');
        container.innerHTML = ''; // Clear previous previews
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const reader = new FileReader();
            reader.onload = function (e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'upload-image';
                container.appendChild(img);
            };
            reader.readAsDataURL(file);
        }
    });
}


function set_error(text='') {
    const errorContainer = document.getElementById('error');
    errorContainer.innerHTML = `${text}`;

}


function call_process(step) {
    // Take all images
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('fileInput', files[i]);
    }
    // Insert other parameters
    formData.append('step', step);

    // Call the server
    $.ajax({
        url: '/new_people',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            console.log("win");
            console.log(response);
            set_error();

            if (response.images) {
                const images = response.images;
                let htmlContent = '';
                images.forEach(image => {
                    htmlContent += `<img src="data:image/jpeg;base64,${image}" alt="Image">`;
                });
                htmlContent += '';
                // Insérer le contenu HTML dans un élément de la page
                document.getElementById('image-container-'+step).innerHTML = htmlContent;
                console.log("ok");
            }


        },
        error: function (error) {
            set_error(error.responseJSON ? error.responseJSON.error : 'Unknown error');
        }
    });
}
