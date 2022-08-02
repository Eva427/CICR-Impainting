inputFile = document.getElementById('inputFile')
tempImg = document.getElementById('tempImg')
predictImg = document.getElementById('predictImg')

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
        contextMask.drawImage(canvasBackMask, 0, 0, imgMask.width, imgMask.height);
        canvasFrontMask.width = imgMask.width;
        canvasFrontMask.height = imgMask.height;

        var dataURL = canvasBackMask.toDataURL("image/png");
        dataURL = dataURL.replace(/^data:image\/(png|jpg);base64,/, "");

        $.post('/imp/segment', 
            {"imgB64" : dataURL}
        ).done(function(response) {
            if (response["ok"]){
                var timestamp = new Date().getTime();  
                var queryString = "?t=" + timestamp;  
                maskPath = "./static/image/mask.jpg" + queryString
                img.src = maskPath ;

                img.onload = function() {
                    var maxSize = Math.max(img.width, img.height);
                    var divisible = Math.round(maxSize / 512);
                    img.width = img.width / divisible;
                    img.height = img.height / divisible;
                    canvasBack.width = img.width;
                    canvasBack.height = img.height;
                    canvasBack.ctx.drawImage(img, 0, 0, img.width, img.height);
                    canvas.width = canvasBack.width;
                    canvas.height = canvasBack.height;
                    context.drawImage(canvasBack, 0, 0, img.width, img.height);
                };
            }
        });
    };
    clearCanvasMask() ;
});

$('#predict').mousedown(function(e) {
    var maskB64 = canvasFrontMask.toDataURL("image/png");
    maskB64 = maskB64.replace(/^data:image\/(png|jpg);base64,/, "");
    var segmentB64 = canvas.toDataURL("image/png");
    segmentB64 = segmentB64.replace(/^data:image\/(png|jpg);base64,/, "");

    $.post('/imp/predict', 
            {"maskB64" : maskB64,
             "segmentB64" : segmentB64}
    ).done(function(response) {
        var timestamp = new Date().getTime();  
        var queryString = "?t=" + timestamp;  
        predictPath = "./static/image/predict.jpg" + queryString;
        predictImg.src = predictPath;
    });
});