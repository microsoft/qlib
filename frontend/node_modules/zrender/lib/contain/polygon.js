import windingLine from './windingLine.js';
var EPSILON = 1e-8;
function isAroundEqual(a, b) {
    return Math.abs(a - b) < EPSILON;
}
export function contain(points, x, y) {
    var w = 0;
    var p = points[0];
    if (!p) {
        return false;
    }
    for (var i = 1; i < points.length; i++) {
        var p2 = points[i];
        w += windingLine(p[0], p[1], p2[0], p2[1], x, y);
        p = p2;
    }
    var p0 = points[0];
    if (!isAroundEqual(p[0], p0[0]) || !isAroundEqual(p[1], p0[1])) {
        w += windingLine(p[0], p[1], p0[0], p0[1], x, y);
    }
    return w !== 0;
}
