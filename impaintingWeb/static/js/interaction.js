inputFile = document.getElementById('inputFile')
tempImg = document.getElementById('tempImg')
predictImg = document.getElementById('predictImg')
enhance = document.getElementById('enhance')
resize = document.getElementById('resize')
updateResize = document.getElementById('updateResize')

loading = document.getElementById('loading')
loading.style.visibility = "hidden";

var firstKeypoints = []

$('#resetKeypoints').mousedown(function(e) {
    resetKeypoints(firstKeypoints);
})

$('#inputFile').change(function(e) {
    objectURL = URL.createObjectURL(inputFile.files[0]);

    // Ajoute à gauche l'image qu'on vient d'importer
    imgMask = new Image();
    imgMask.src = objectURL;

    imgMask.onload = function() {
        var maxSizeMask = Math.max(imgMask.width, imgMask.height);
        var divisibleMask = Math.round(maxSizeMask / 450);
        imgMask.width = imgMask.width / divisibleMask;
        imgMask.height = imgMask.height / divisibleMask;
        canvasBackMask.width = imgMask.width;
        canvasBackMask.height = imgMask.height;
        canvasBackMask.ctx.drawImage(imgMask, 0, 0, imgMask.width, imgMask.height);
        canvasMask.width = imgMask.width;
        canvasMask.height = imgMask.height;
        contextMask.drawImage(canvasBackMask, 0, 0);
        canvasFrontMask.width = imgMask.width;
        canvasFrontMask.height = imgMask.height;

        var dataURL = canvasBackMask.toDataURL("image/png");
        dataURL = dataURL.replace(/^data:image\/(png|jpg);base64,/, "");

        // Envoie au backend l'image sous forme de texte encodé puis la trace au milieu pour la segmentation
        // La réponse est censé être un vecteur de coordonée (pour les keypoints)
        $.post('/imp/segment', { "imgB64": dataURL, "size": resize.value}).done(function(response) {
            if (response) {
                var timestamp = new Date().getTime();
                var queryString = "?t=" + timestamp; // permet de refresh le src automatiquement

                imgBack = new Image();
                imgMiddle = new Image();

                imgBack.src = "./static/image/mask.jpg" + queryString;
                imgMiddle.src = "./static/image/original_crop.jpg" + queryString;

                imgBack.onload = function() {
                    imgMiddle.onload = function() {
                        var maxSize = Math.max(imgBack.width, imgBack.height);
                        var divisible = Math.round(maxSize / 512);
                        divisible = Math.max(divisible, 1);
                        imgBack.width = imgBack.width / divisible;
                        imgBack.height = imgBack.height / divisible;
                        canvasBack.width = imgBack.width;
                        canvasBack.height = imgBack.height;
                        canvasBack.ctx.drawImage(imgBack, 0, 0, imgBack.width, imgBack.height);
                        canvasMiddle.width = imgBack.width;
                        canvasMiddle.height = imgBack.height
                        canvasMiddle.ctx.drawImage(imgMiddle, 0, 0, imgBack.width, imgBack.height);
                        canvas.width = canvasBack.width;
                        canvas.height = canvasBack.height;
                        context.drawImage(canvasBack, 0, 0);
                        context.globalAlpha = 0.2;
                        context.drawImage(canvasMiddle, 0, 0);
                        context.globalAlpha = 1;
                        canvasFront.width = imgBack.width;
                        canvasFront.height = imgBack.height;
                    }
                };

                // Keypoints
                imageObj.src = "./static/image/original_crop.jpg" + queryString;
                firstKeypoints = response["keypoints"]
                removeKeypoints()
                addAllKeypoints(response["keypoints"]);
            }
        });
    };
    clearCanvasMask();
});

$('#updateResize').mousedown(function(e) {

    var dataURL = canvasBackMask.toDataURL("image/png");
    dataURL = dataURL.replace(/^data:image\/(png|jpg);base64,/, "");

    $.post('/imp/segment', { "imgB64": dataURL, "size": resize.value}).done(function(response) {
        if (response) {
            var timestamp = new Date().getTime();
            var queryString = "?t=" + timestamp; // permet de refresh le src automatiquement

            imgBack = new Image();
            imgMiddle = new Image();

            imgBack.src = "./static/image/mask.jpg" + queryString;
            imgMiddle.src = "./static/image/original_crop.jpg" + queryString;

            imgBack.onload = function() {
                imgMiddle.onload = function() {
                    var maxSize = Math.max(imgBack.width, imgBack.height);
                    var divisible = Math.round(maxSize / 512);
                    divisible = Math.max(divisible, 1);
                    imgBack.width = imgBack.width / divisible;
                    imgBack.height = imgBack.height / divisible;
                    canvasBack.width = imgBack.width;
                    canvasBack.height = imgBack.height;
                    canvasBack.ctx.drawImage(imgBack, 0, 0, imgBack.width, imgBack.height);
                    canvasMiddle.width = imgBack.width;
                    canvasMiddle.height = imgBack.height
                    canvasMiddle.ctx.drawImage(imgMiddle, 0, 0, imgBack.width, imgBack.height);
                    canvas.width = canvasBack.width;
                    canvas.height = canvasBack.height;
                    context.drawImage(canvasBack, 0, 0);
                    context.globalAlpha = 0.2;
                    context.drawImage(canvasMiddle, 0, 0);
                    context.globalAlpha = 1;
                    canvasFront.width = imgBack.width;
                    canvasFront.height = imgBack.height;
                }
            };

            // Keypoints
            imageObj.src = "./static/image/original_crop.jpg" + queryString;
            firstKeypoints = response["keypoints"]
            removeKeypoints()
            addAllKeypoints(response["keypoints"]);
        }
    });
});

$('#predict').mousedown(function(e) {
    loading.style.visibility = "visible";
    var maskB64 = canvasFrontMask.toDataURL("image/png");
    maskB64 = maskB64.replace(/^data:image\/(png|jpg);base64,/, "");

    redraw(false);
    var segmentB64 = canvas.toDataURL("image/png");
    segmentB64 = segmentB64.replace(/^data:image\/(png|jpg);base64,/, "");
    redraw();

    $.post('/imp/predict', {
        "maskB64": maskB64,
        "segmentB64": segmentB64,
        "doEnhance": enhance.checked,
        "keypoints": JSON.stringify(getAllKeypoints()),
        "modelOpt": document.querySelector('input[name="modelOption"]:checked').value
    }).done(function(response) {
        loading.style.visibility = "hidden";
        var timestamp = new Date().getTime();
        var queryString = "?t=" + timestamp;
        predictPath = "./static/image/predict.jpg" + queryString;
        predictImg.src = predictPath;
    });
});

$('#download').mousedown(function(e) {
    var maskB64 = canvasFrontMask.toDataURL("image/png");
    maskB64 = maskB64.replace(/^data:image\/(png|jpg);base64,/, "");

    redraw(false);
    var segmentB64 = canvas.toDataURL("image/png");
    segmentB64 = segmentB64.replace(/^data:image\/(png|jpg);base64,/, "");
    redraw();
    var filename = inputFile.files[0].name;
    filename = filename.split('.')[0];
    var url = "./static/image/downloaded/"

    $.post('/imp/download', {
        "maskB64": maskB64,
        "segmentB64": segmentB64
    }).done(function(response) {
        var link = document.createElement('a');
        document.body.appendChild(link);

        link.href = url + "mask.jpg";
        link.download = filename + "_mask.jpg";
        link.click();

        link.href = url + "seg.jpg";
        link.download = filename + "_seg.jpg";
        link.click();

        document.body.removeChild(link);
    });
});