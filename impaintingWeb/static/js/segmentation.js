canvas = document.getElementById('canvas')
context = canvas.getContext("2d");

var canvasBack = document.createElement("canvas");
canvasBack.ctx = canvasBack.getContext("2d");
var canvasMiddle = document.createElement("canvas");
canvasMiddle.ctx = canvasMiddle.getContext("2d");
var canvasFront = document.createElement("canvas");
canvasFront.ctx = canvasFront.getContext("2d");

var imgBack = new Image();
var imgMiddle = new Image();

$('#canvas').mousedown(function(e) {
    paint = true;
    addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
    redraw();
});

$('#canvas').mousemove(function(e) {
    if (paint) {
        addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
        redraw();
    }
});
$('#canvas').mouseup(function(e) {
    paint = false;
});
$('#canvas').mouseleave(function(e) {
    paint = false;
});

var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var clickColor = new Array();
var clickSize = new Array();
var paint;

var curColor = "#191919";
var curSize = 10;

function addClick(x, y, dragging) {
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
    clickColor.push(curColor);
    clickSize.push(curSize);
}

function undoClick() {
    clickDrag.pop();
    clickX.pop();
    clickY.pop();
    clickColor.pop();
    clickSize.pop();
    redraw();
}

function clearCanvas() {
    clickX = new Array();
    clickY = new Array();
    clickDrag = new Array();
    clickColor = new Array();
    clickSize = new Array();
    redraw();
}

function redraw(calque = true) {
    canvasFront.ctx.lineJoin = "round";
    canvasFront.ctx.clearRect(0, 0, canvasFront.ctx.canvas.width, canvasFront.ctx.canvas.height); // Clears the canvas
    for (var i = 0; i < clickX.length; i++) {
        canvasFront.ctx.beginPath();
        if (clickDrag[i] && i) {
            canvasFront.ctx.moveTo(clickX[i - 1], clickY[i - 1]);
        } else {
            canvasFront.ctx.moveTo(clickX[i] - 1, clickY[i]);
        }
        canvasFront.ctx.lineTo(clickX[i], clickY[i]);
        canvasFront.ctx.closePath();
        canvasFront.ctx.strokeStyle = clickColor[i];
        canvasFront.ctx.lineWidth = clickSize[i];
        canvasFront.ctx.stroke();
    }
    context.drawImage(canvasBack, 0, 0);
    context.drawImage(canvasFront, 0, 0);
    if (calque) {
        context.globalAlpha = 0.2;
        context.drawImage(canvasMiddle, 0, 0);
        context.globalAlpha = 1;
    }
}

$('#chooseColorBG').mousedown(function(e) {
    curColor = "#191919";
});
$('#chooseColorSkin').mousedown(function(e) {
    curColor = "#323232";
});
$('#chooseColorNose').mousedown(function(e) {
    curColor = "#4B4B4B";
});
$('#chooseColorEye').mousedown(function(e) {
    curColor = "#646464";
});
$('#chooseColorEyebrow').mousedown(function(e) {
    curColor = "#7D7D7D";
});
$('#chooseColorEar').mousedown(function(e) {
    curColor = "#969696";
});
$('#chooseColorMouth').mousedown(function(e) {
    curColor = "#AFAFAF";
});
$('#chooseColorLip').mousedown(function(e) {
    curColor = "#C8C8C8";
});


$('#clear').mousedown(function(e) {
    clearCanvas();
});

$('#size').change(function(e) {
    curSize = this.value;
});

$('#undo').mousedown(function(e) {
    if (!clickDrag[clickDrag.length - 1]) {
        undoClick()
    } else {
        while (clickDrag[clickDrag.length - 1]) {
            undoClick()
        }
        undoClick()
    }
});