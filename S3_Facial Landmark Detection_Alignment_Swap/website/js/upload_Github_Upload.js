
function uploadAndClassifyImage(){
    var fileInput = document.getElementById('mobilenetFileUpload').files;
    if(!fileInput.length){
        return alert('Please choose a file to upload first. ');
    }

    var file = fileInput[0];
    var filename = file.name

    var formData = new FormData();
    formData.append(filename, file);

    console.log(filename);

    $.ajax({
        async: true,
        crossDomain: true,
        method: 'POST',
        url: 'https://1abcd234ef.execute-api.ap-south-1.amazonaws.com/dev/classify',
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data",
    })
    .done(function (response) {
        console.log(response);
        var json = $.parseJSON(response)
        document.getElementById('result').textContent = json['predicted'];
    })
    .fail(function(){
        alert("There was an error while sending prediction request to mobilenet model.");
    });
};


$('#btnMobileNetUpload').click(uploadAndClassifyImage);





function uploadAndClassifyFlyingObject(){
    var fileInput = document.getElementById('mobilenetFileUpload').files;
    if(!fileInput.length){
        return alert('Please choose a file to upload first. ');
    }

    var file = fileInput[0];
    var filename = file.name

    var formData = new FormData();
    formData.append(filename, file);

    console.log(filename);

    $.ajax({
        async: true,
        crossDomain: true,
        method: 'POST',
        url: 'https://xyz1stuv23.execute-api.ap-south-1.amazonaws.com/dev/classify',
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data",
    })
    .done(function (response) {
        console.log(response);
        var json = $.parseJSON(response)
        document.getElementById('result').textContent = json['predicted class'];
    })
    .fail(function(){
        alert("There was an error while sending prediction request to mobilenet model.");
    });
};



$('#btnFlyingObjects').click(uploadAndClassifyFlyingObject);





function uploadAndAlignImage(){
    var fileInput = document.getElementById('faceAlignFileUpload').files;
    if(!fileInput.length){
        return alert('Please choose a file to upload first. ');
    }

    var file = fileInput[0];
    var filename = file.name

    

    var formData = new FormData();
    formData.append(filename, file);

    console.log(filename);

    $.ajax({
        async: true,
        crossDomain: true,
        method: 'POST',
        url: 'https://lmnop7qrst.execute-api.ap-south-1.amazonaws.com/dev/align',
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data",
    })
    .done(function (response) {
        var json = $.parseJSON(response)
        document.getElementById('resimage').src = json['alignedFaceImg']
        console.log(response);
        
    })
    .fail(function(){
        alert("There was an error while sending request to face aligner.");
    });
};
$('#btnFaceAlignUpload').click(uploadAndAlignImage);
