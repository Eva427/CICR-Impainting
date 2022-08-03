canvasMask = document.getElementById('canvasMask')
contextMask = canvasMask.getContext("2d");
var canvasBackMask = document.createElement("canvas");
canvasBackMask.ctx = canvasBackMask.getContext("2d");
var canvasFrontMask = document.createElement("canvas");
canvasFrontMask.ctx = canvasFrontMask.getContext("2d");
var imgMask = new Image();

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
    canvasFrontMask.ctx.save();
    canvasFrontMask.ctx.lineJoin = "round";
    canvasFrontMask.ctx.clearRect(0, 0, canvasFrontMask.ctx.canvas.width, canvasFrontMask.ctx.canvas.height); // Clears the canvasMask
    canvasFrontMask.ctx.strokeStyle = curColorMask;

    for (var i = 0; i < clickXMask.length; i++) {
        canvasFrontMask.ctx.beginPath();
        if (clickDragMask[i] && i) {
            canvasFrontMask.ctx.moveTo(clickXMask[i - 1], clickYMask[i - 1]);
        } else {
            canvasFrontMask.ctx.moveTo(clickXMask[i] - 1, clickYMask[i]);
        }
        canvasFrontMask.ctx.lineTo(clickXMask[i], clickYMask[i]);
        canvasFrontMask.ctx.closePath();
        canvasFrontMask.ctx.lineWidth = clickSizeMask[i];
        canvasFrontMask.ctx.stroke();
    }

    contextMask.drawImage(canvasBackMask, 0, 0);
    contextMask.drawImage(canvasFrontMask, 0, 0);
    canvasFrontMask.ctx.restore();
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