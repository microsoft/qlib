import { fromPoints } from '../core/bbox.js';
import BoundingRect from '../core/BoundingRect.js';
import Point from '../core/Point.js';
import { map } from '../core/util.js';
import Polygon from '../graphic/shape/Polygon.js';
import Rect from '../graphic/shape/Rect.js';
import Sector from '../graphic/shape/Sector.js';
import { pathToPolygons } from './convertPath.js';
import { clonePath } from './path.js';
function getDividingGrids(dimSize, rowDim, count) {
    var rowSize = dimSize[rowDim];
    var columnSize = dimSize[1 - rowDim];
    var ratio = Math.abs(rowSize / columnSize);
    var rowCount = Math.ceil(Math.sqrt(ratio * count));
    var columnCount = Math.floor(count / rowCount);
    if (columnCount === 0) {
        columnCount = 1;
        rowCount = count;
    }
    var grids = [];
    for (var i = 0; i < rowCount; i++) {
        grids.push(columnCount);
    }
    var currentCount = rowCount * columnCount;
    var remained = count - currentCount;
    if (remained > 0) {
        for (var i = 0; i < remained; i++) {
            grids[i % rowCount] += 1;
        }
    }
    return grids;
}
function divideSector(sectorShape, count, outShapes) {
    var r0 = sectorShape.r0;
    var r = sectorShape.r;
    var startAngle = sectorShape.startAngle;
    var endAngle = sectorShape.endAngle;
    var angle = Math.abs(endAngle - startAngle);
    var arcLen = angle * r;
    var deltaR = r - r0;
    var isAngleRow = arcLen > Math.abs(deltaR);
    var grids = getDividingGrids([arcLen, deltaR], isAngleRow ? 0 : 1, count);
    var rowSize = (isAngleRow ? angle : deltaR) / grids.length;
    for (var row = 0; row < grids.length; row++) {
        var columnSize = (isAngleRow ? deltaR : angle) / grids[row];
        for (var column = 0; column < grids[row]; column++) {
            var newShape = {};
            if (isAngleRow) {
                newShape.startAngle = startAngle + rowSize * row;
                newShape.endAngle = startAngle + rowSize * (row + 1);
                newShape.r0 = r0 + columnSize * column;
                newShape.r = r0 + columnSize * (column + 1);
            }
            else {
                newShape.startAngle = startAngle + columnSize * column;
                newShape.endAngle = startAngle + columnSize * (column + 1);
                newShape.r0 = r0 + rowSize * row;
                newShape.r = r0 + rowSize * (row + 1);
            }
            newShape.clockwise = sectorShape.clockwise;
            newShape.cx = sectorShape.cx;
            newShape.cy = sectorShape.cy;
            outShapes.push(newShape);
        }
    }
}
function divideRect(rectShape, count, outShapes) {
    var width = rectShape.width;
    var height = rectShape.height;
    var isHorizontalRow = width > height;
    var grids = getDividingGrids([width, height], isHorizontalRow ? 0 : 1, count);
    var rowSizeDim = isHorizontalRow ? 'width' : 'height';
    var columnSizeDim = isHorizontalRow ? 'height' : 'width';
    var rowDim = isHorizontalRow ? 'x' : 'y';
    var columnDim = isHorizontalRow ? 'y' : 'x';
    var rowSize = rectShape[rowSizeDim] / grids.length;
    for (var row = 0; row < grids.length; row++) {
        var columnSize = rectShape[columnSizeDim] / grids[row];
        for (var column = 0; column < grids[row]; column++) {
            var newShape = {};
            newShape[rowDim] = row * rowSize;
            newShape[columnDim] = column * columnSize;
            newShape[rowSizeDim] = rowSize;
            newShape[columnSizeDim] = columnSize;
            newShape.x += rectShape.x;
            newShape.y += rectShape.y;
            outShapes.push(newShape);
        }
    }
}
function crossProduct2d(x1, y1, x2, y2) {
    return x1 * y2 - x2 * y1;
}
function lineLineIntersect(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y) {
    var mx = a2x - a1x;
    var my = a2y - a1y;
    var nx = b2x - b1x;
    var ny = b2y - b1y;
    var nmCrossProduct = crossProduct2d(nx, ny, mx, my);
    if (Math.abs(nmCrossProduct) < 1e-6) {
        return null;
    }
    var b1a1x = a1x - b1x;
    var b1a1y = a1y - b1y;
    var p = crossProduct2d(b1a1x, b1a1y, nx, ny) / nmCrossProduct;
    if (p < 0 || p > 1) {
        return null;
    }
    return new Point(p * mx + a1x, p * my + a1y);
}
function projPtOnLine(pt, lineA, lineB) {
    var dir = new Point();
    Point.sub(dir, lineB, lineA);
    dir.normalize();
    var dir2 = new Point();
    Point.sub(dir2, pt, lineA);
    var len = dir2.dot(dir);
    return len;
}
function addToPoly(poly, pt) {
    var last = poly[poly.length - 1];
    if (last && last[0] === pt[0] && last[1] === pt[1]) {
        return;
    }
    poly.push(pt);
}
function splitPolygonByLine(points, lineA, lineB) {
    var len = points.length;
    var intersections = [];
    for (var i = 0; i < len; i++) {
        var p0 = points[i];
        var p1 = points[(i + 1) % len];
        var intersectionPt = lineLineIntersect(p0[0], p0[1], p1[0], p1[1], lineA.x, lineA.y, lineB.x, lineB.y);
        if (intersectionPt) {
            intersections.push({
                projPt: projPtOnLine(intersectionPt, lineA, lineB),
                pt: intersectionPt,
                idx: i
            });
        }
    }
    if (intersections.length < 2) {
        return [{ points: points }, { points: points }];
    }
    intersections.sort(function (a, b) {
        return a.projPt - b.projPt;
    });
    var splitPt0 = intersections[0];
    var splitPt1 = intersections[intersections.length - 1];
    if (splitPt1.idx < splitPt0.idx) {
        var tmp = splitPt0;
        splitPt0 = splitPt1;
        splitPt1 = tmp;
    }
    var splitPt0Arr = [splitPt0.pt.x, splitPt0.pt.y];
    var splitPt1Arr = [splitPt1.pt.x, splitPt1.pt.y];
    var newPolyA = [splitPt0Arr];
    var newPolyB = [splitPt1Arr];
    for (var i = splitPt0.idx + 1; i <= splitPt1.idx; i++) {
        addToPoly(newPolyA, points[i].slice());
    }
    addToPoly(newPolyA, splitPt1Arr);
    addToPoly(newPolyA, splitPt0Arr);
    for (var i = splitPt1.idx + 1; i <= splitPt0.idx + len; i++) {
        addToPoly(newPolyB, points[i % len].slice());
    }
    addToPoly(newPolyB, splitPt0Arr);
    addToPoly(newPolyB, splitPt1Arr);
    return [{
            points: newPolyA
        }, {
            points: newPolyB
        }];
}
function binaryDividePolygon(polygonShape) {
    var points = polygonShape.points;
    var min = [];
    var max = [];
    fromPoints(points, min, max);
    var boundingRect = new BoundingRect(min[0], min[1], max[0] - min[0], max[1] - min[1]);
    var width = boundingRect.width;
    var height = boundingRect.height;
    var x = boundingRect.x;
    var y = boundingRect.y;
    var pt0 = new Point();
    var pt1 = new Point();
    if (width > height) {
        pt0.x = pt1.x = x + width / 2;
        pt0.y = y;
        pt1.y = y + height;
    }
    else {
        pt0.y = pt1.y = y + height / 2;
        pt0.x = x;
        pt1.x = x + width;
    }
    return splitPolygonByLine(points, pt0, pt1);
}
function binaryDivideRecursive(divider, shape, count, out) {
    if (count === 1) {
        out.push(shape);
    }
    else {
        var mid = Math.floor(count / 2);
        var sub = divider(shape);
        binaryDivideRecursive(divider, sub[0], mid, out);
        binaryDivideRecursive(divider, sub[1], count - mid, out);
    }
    return out;
}
export function clone(path, count) {
    var paths = [];
    for (var i = 0; i < count; i++) {
        paths.push(clonePath(path));
    }
    return paths;
}
function copyPathProps(source, target) {
    target.setStyle(source.style);
    target.z = source.z;
    target.z2 = source.z2;
    target.zlevel = source.zlevel;
}
function polygonConvert(points) {
    var out = [];
    for (var i = 0; i < points.length;) {
        out.push([points[i++], points[i++]]);
    }
    return out;
}
export function split(path, count) {
    var outShapes = [];
    var shape = path.shape;
    var OutShapeCtor;
    switch (path.type) {
        case 'rect':
            divideRect(shape, count, outShapes);
            OutShapeCtor = Rect;
            break;
        case 'sector':
            divideSector(shape, count, outShapes);
            OutShapeCtor = Sector;
            break;
        case 'circle':
            divideSector({
                r0: 0, r: shape.r, startAngle: 0, endAngle: Math.PI * 2,
                cx: shape.cx, cy: shape.cy
            }, count, outShapes);
            OutShapeCtor = Sector;
            break;
        default:
            var m = path.getComputedTransform();
            var scale = m ? Math.sqrt(Math.max(m[0] * m[0] + m[1] * m[1], m[2] * m[2] + m[3] * m[3])) : 1;
            var polygons = map(pathToPolygons(path.getUpdatedPathProxy(), scale), function (poly) { return polygonConvert(poly); });
            var polygonCount = polygons.length;
            if (polygonCount === 0) {
                binaryDivideRecursive(binaryDividePolygon, {
                    points: polygons[0]
                }, count, outShapes);
            }
            else if (polygonCount === count) {
                for (var i = 0; i < polygonCount; i++) {
                    outShapes.push({
                        points: polygons[i]
                    });
                }
            }
            else {
                var totalArea_1 = 0;
                var items = map(polygons, function (poly) {
                    var min = [];
                    var max = [];
                    fromPoints(poly, min, max);
                    var area = (max[1] - min[1]) * (max[0] - min[0]);
                    totalArea_1 += area;
                    return { poly: poly, area: area };
                });
                items.sort(function (a, b) { return b.area - a.area; });
                var left = count;
                for (var i = 0; i < polygonCount; i++) {
                    var item = items[i];
                    if (left <= 0) {
                        break;
                    }
                    var selfCount = i === polygonCount - 1
                        ? left
                        : Math.ceil(item.area / totalArea_1 * count);
                    if (selfCount < 0) {
                        continue;
                    }
                    binaryDivideRecursive(binaryDividePolygon, {
                        points: item.poly
                    }, selfCount, outShapes);
                    left -= selfCount;
                }
                ;
            }
            OutShapeCtor = Polygon;
            break;
    }
    if (!OutShapeCtor) {
        return clone(path, count);
    }
    var out = [];
    for (var i = 0; i < outShapes.length; i++) {
        var subPath = new OutShapeCtor();
        subPath.setShape(outShapes[i]);
        copyPathProps(path, subPath);
        out.push(subPath);
    }
    return out;
}
