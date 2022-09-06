var width = 256;
var height = 256;

var stage = new Konva.Stage({
    container: 'KonvaContainer',
    width: width,
    height: height,
});

var layer1 = new Konva.Layer();
var layer2 = new Konva.Layer();

var imageObj = new Image();
imageObj.onload = function() {
    var yoda = new Konva.Image({
        image: imageObj,
        width: width,
        height: height,
    });
    layer1.add(yoda);
};

var pointList = [];

var createKeypoint = function(x, y) {
    var point = new Konva.Circle({
        x: x,
        y: y,
        radius: 4,
        fill: '#00D2FF',
        stroke: 'black',
        strokeWidth: 1.5,
        draggable: true,
    });
    pointList.push(point)

    // add cursor styling
    point.on('mouseover', function() {
        document.body.style.cursor = 'pointer';
    });
    point.on('mouseout', function() {
        document.body.style.cursor = 'default';
    });
    layer2.add(point);
};

var addAllKeypoints = function(keypoints) {
    for (const [x, y] of keypoints) {
        createKeypoint(x * 2, y * 2);
    }
}

var getAllKeypoints = function() {
    var coordinates = [];
    for (const point of pointList) {
        coordinates.push([point.attrs.x / 2, point.attrs.y / 2])
    }
    return coordinates
}

var resetKeypoints = function(keypoints) {
    layer2.destroyChildren();
    pointList = [];
    addAllKeypoints(keypoints);
}

// addAllKeypoints(keypoints);
// console.log(pointList);

stage.add(layer1);
stage.add(layer2);