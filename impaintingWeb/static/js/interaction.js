inputFile = document.getElementById('inputFile')
tempImg = document.getElementById('tempImg')
predictImg = document.getElementById('predictImg')
enhance = document.getElementById('enhance')

loading = document.getElementById('loading')
loading.style.visibility = "hidden";

$('#inputFile').change(function(e) {
    objectURL = URL.createObjectURL(inputFile.files[0]);
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

        $.post('/imp/segment', { "imgB64": dataURL }).done(function(response) {
            if (response["ok"]) {
                var timestamp = new Date().getTime();
                var queryString = "?t=" + timestamp;
                imgBack.src = "./static/image/mask.jpg" + queryString
                imgMiddle.src = "./static/image/original_crop.jpg" + queryString

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
            }
        });
    };
    clearCanvasMask();
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
        "doEnhance": enhance.checked
    }).done(function(response) {
        loading.style.visibility = "hidden";
        var timestamp = new Date().getTime();
        var queryString = "?t=" + timestamp;
        predictPath = "./static/image/predict.jpg" + queryString;
        predictImg.src = predictPath;
    });
});