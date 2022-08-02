canvas = document.getElementById('canvas')
context = canvas.getContext("2d");
var canvasBack = document.createElement("canvas");
canvasBack.ctx = canvasBack.getContext("2d");
var img = new Image();


$('#canvas').mousedown(function(e) {
    var mouseX = e.pageX - this.offsetLeft;
    var mouseY = e.pageY - this.offsetTop;

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

function redraw() {
    context.save();
    context.lineJoin = "round";
    context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
    context.drawImage(canvasBack, 0, 0);
    for (var i = 0; i < clickX.length; i++) {
        context.beginPath();
        if (clickDrag[i] && i) {
            context.moveTo(clickX[i - 1], clickY[i - 1]);
        } else {
            context.moveTo(clickX[i] - 1, clickY[i]);
        }
        context.lineTo(clickX[i], clickY[i]);
        context.closePath();
        context.strokeStyle = clickColor[i];
        context.lineWidth = clickSize[i];
        context.stroke();
    }
    context.restore();
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