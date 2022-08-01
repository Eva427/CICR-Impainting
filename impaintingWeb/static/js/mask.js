canvasMask = document.getElementById('canvasMask')
contextMask = canvasMask.getContext("2d");
var canvasBackMask = document.createElement("canvas");
canvasBackMask.ctx = canvasBackMask.getContext("2d");

var imgMask = new Image();
imgMask.onload = function() {

    var maxSizeMask = Math.max(imgMask.width, imgMask.height);
    var divisibleMask = Math.round(maxSizeMask / 512);
    imgMask.width = imgMask.width / divisibleMask;
    imgMask.height = imgMask.height / divisibleMask;

    canvasBackMask.width = imgMask.width;
    canvasBackMask.height = imgMask.height;
    canvasBackMask.ctx.drawImage(imgMask, 0, 0, imgMask.width, imgMask.height);
    canvasMask.width = canvasBackMask.width;
    canvasMask.height = canvasBackMask.height;
    contextMask.drawImage(canvasBackMask, 0, 0, imgMask.width, imgMask.height);
};
imgMask.src = './static/image/input.jpg';

$('#canvasMask').mousedown(function(e) {
    paintMask = true;
    addClickMask(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
    redrawMask();
});

$('#canvasMask').mousemove(function(e) {
    if (paintMask) {
        addClickMask(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
        redrawMask();
    }
});
$('#canvasMask').mouseup(function(e) {
    paintMask = false;
});
$('#canvasMask').mouseleave(function(e) {
    paintMask = false;
});

var clickXMask = new Array();
var clickYMask = new Array();
var clickDragMask = new Array();
var clickSizeMask = new Array();
var paintMask;

var curColorMask = "#000000";
var curSizeMask = 10;

function addClickMask(x, y, dragging) {
    clickXMask.push(x);
    clickYMask.push(y);
    clickDragMask.push(dragging);
    clickSizeMask.push(curSizeMask);
}

function undoClickMask() {
    clickDragMask.pop();
    clickXMask.pop();
    clickYMask.pop();
    clickSizeMask.pop();
    redrawMask();
}

function clearCanvasMask() {
    clickXMask = new Array();
    clickYMask = new Array();
    clickDragMask = new Array();
    clickSizeMask = new Array();
    redrawMask();
}

function redrawMask() {
    contextMask.save();
    contextMask.lineJoin = "round";
    contextMask.clearRect(0, 0, contextMask.canvas.width, contextMask.canvas.height); // Clears the canvasMask
    contextMask.strokeStyle = curColorMask;
    contextMask.drawImage(canvasBackMask, 0, 0);
    for (var i = 0; i < clickXMask.length; i++) {
        contextMask.beginPath();
        if (clickDragMask[i] && i) {
            contextMask.moveTo(clickXMask[i - 1], clickYMask[i - 1]);
        } else {
            contextMask.moveTo(clickXMask[i] - 1, clickYMask[i]);
        }
        contextMask.lineTo(clickXMask[i], clickYMask[i]);
        contextMask.closePath();
        contextMask.lineWidth = clickSizeMask[i];
        contextMask.stroke();
    }
    contextMask.restore();
}

$('#clearMask').mousedown(function(e) {
    clearCanvasMask();
});

$('#sizeMask').change(function(e) {
    curSizeMask = this.value;
});

$('#undoMask').mousedown(function(e) {
    if (!clickDragMask[clickDragMask.length - 1]) {
        undoClickMask()
    } else {
        while (clickDragMask[clickDragMask.length - 1]) {
            undoClickMask()
        }
        undoClickMask()
    }
});