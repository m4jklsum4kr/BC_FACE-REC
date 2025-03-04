get_database();

function get_database() {
    $.ajax({
        url: '/api/db_search_all',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({}),
        success: function(data) {
            // Traitez le résultat ici
            const res = document.getElementById('db_all')
            console.log(data)
            res.innerHTML = "";
            for (let i = 0; i < data.result.length; i++) {
                res.innerHTML += `<option value="${data.result[i]}">${data.result[i]}</option>\n`;
            }

        },
        error: function(error) {console.error('Erreur Jquery:', error);}
    });
}


function change_image(image_path) {
    let image = document.getElementById('image');
    image.src = "{{ url_for('static', filename='..\\..\\"+image_path+"') }}";
    console.log('change img')
}


document.addEventListener('DOMContentLoaded', function () {

    // SELECT_BOX: db_all
    const selectBox_db_all = document.getElementById('db_all');
    let previousValue = null;
    selectBox_db_all.addEventListener('change', function () {
        const selectedValue = selectBox_db_all.value;
        if (selectedValue !== previousValue) {
            console.log('Selected value:', selectedValue);
            change_image(selectedValue);
            previousValue = selectedValue;
        }
    });

});


