import { fromPoints } from '../core/bbox';
import BoundingRect from '../core/BoundingRect';
import Point from '../core/Point';
import { map } from '../core/util';
import Path from '../graphic/Path';
import Polygon from '../graphic/shape/Polygon';
import Rect from '../graphic/shape/Rect';
import Sector from '../graphic/shape/Sector';
import { pathToPolygons } from './convertPath';
import { clonePath } from './path';

// Default shape dividers
// TODO divide polygon by grids.
interface BinaryDivide {
    (shape: Path['shape']): Path['shape'][]
}

/**
 * Calculating a grid to divide the shape.
 */
function getDividingGrids(dimSize: number[], rowDim: number, count: number) {
    const rowSize = dimSize[rowDim];
    const columnSize = dimSize[1 - rowDim];

    const ratio = Math.abs(rowSize / columnSize);
    let rowCount = Math.ceil(Math.sqrt(ratio * count));
    let columnCount = Math.floor(count / rowCount);
    if (columnCount === 0) {
        columnCount = 1;
        rowCount = count;
    }

    const grids: number[] = [];
    for (let i = 0; i < rowCount; i++) {
        grids.push(columnCount);
    }
    const currentCount = rowCount * columnCount;
    // Distribute the remaind grid evenly on each row.
    const remained = count - currentCount;
    if (remained > 0) {
        // const stride = Math.max(Math.floor(rowCount / remained), 1);
        for (let i = 0; i < remained; i++) {
            grids[i % rowCount] += 1;
        }
    }
    return grids;
}


// TODO cornerRadius
function divideSector(sectorShape: Sector['shape'], count: number, outShapes: Sector['shape'][]) {
    const r0 = sectorShape.r0;
    const r = sectorShape.r;
    const startAngle = sectorShape.startAngle;
    const endAngle = sectorShape.endAngle;
    const angle = Math.abs(endAngle - startAngle);
    const arcLen = angle * r;
    const deltaR = r - r0;

    const isAngleRow = arcLen > Math.abs(deltaR);
    const grids = getDividingGrids([arcLen, deltaR], isAngleRow ? 0 : 1, count);

    const rowSize = (isAngleRow ? angle : deltaR) / grids.length;

    for (let row = 0; row < grids.length; row++) {
        const columnSize = (isAngleRow ? deltaR : angle) / grids[row];
        for (let column = 0; column < grids[row]; column++) {
            const newShape = {} as Sector['shape'];

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

function divideRect(rectShape: Rect['shape'], count: number, outShapes: Rect['shape'][]) {
    const width = rectShape.width;
    const height = rectShape.height;

    const isHorizontalRow = width > height;
    const grids = getDividingGrids([width, height], isHorizontalRow ? 0 : 1, count);
    const rowSizeDim = isHorizontalRow ? 'width' : 'height';
    const columnSizeDim = isHorizontalRow ? 'height' : 'width';
    const rowDim = isHorizontalRow ? 'x' : 'y';
    const columnDim = isHorizontalRow ? 'y' : 'x';
    const rowSize = rectShape[rowSizeDim] / grids.length;

    for (let row = 0; row < grids.length; row++) {
        const columnSize = rectShape[columnSizeDim] / grids[row];
        for (let column = 0; column < grids[row]; column++) {
            const newShape = {} as Rect['shape'];
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

function crossProduct2d(x1: number, y1: number, x2: number, y2: number) {
    return x1 * y2 - x2 * y1;
}

function lineLineIntersect(
    a1x: number, a1y: number, a2x: number, a2y: number, // p1
    b1x: number, b1y: number, b2x: number, b2y: number // p2
): Point {
    const mx = a2x - a1x;
    const my = a2y - a1y;
    const nx = b2x - b1x;
    const ny = b2y - b1y;

    const nmCrossProduct = crossProduct2d(nx, ny, mx, my);
    if (Math.abs(nmCrossProduct) < 1e-6) {
        return null;
    }

    const b1a1x = a1x - b1x;
    const b1a1y = a1y - b1y;

    const p = crossProduct2d(b1a1x, b1a1y, nx, ny) / nmCrossProduct;
    if (p < 0 || p > 1) {
        return null;
    }
    // p2 is an infinite line
    return new Point(
        p * mx + a1x,
        p * my + a1y
    );
}

function projPtOnLine(pt: Point, lineA: Point, lineB: Point): number {
    const dir = new Point();
    Point.sub(dir, lineB, lineA);
    dir.normalize();
    const dir2 = new Point();
    Point.sub(dir2, pt, lineA);
    const len = dir2.dot(dir);
    return len;
}

function addToPoly(poly: number[][], pt: number[]) {
    const last = poly[poly.length - 1];
    if (last && last[0] === pt[0] && last[1] === pt[1]) {
        return;
    }
    poly.push(pt);
}

function splitPolygonByLine(points: number[][], lineA: Point, lineB: Point) {
    const len = points.length;
    const intersections: {
        projPt: number,
        pt: Point
        idx: number
    }[] = [];
    for (let i = 0; i < len; i++) {
        const p0 = points[i];
        const p1 = points[(i + 1) % len];
        const intersectionPt = lineLineIntersect(
            p0[0], p0[1], p1[0], p1[1],
            lineA.x, lineA.y, lineB.x, lineB.y
        );
        if (intersectionPt) {
            intersections.push({
                projPt: projPtOnLine(intersectionPt, lineA, lineB),
                pt: intersectionPt,
                idx: i
            });
        }
    }

    // TODO No intersection?
    if (intersections.length < 2) {
        // Do clone
        return [ { points}, {points} ];
    }

    // Find two farthest points.
    intersections.sort((a, b) => {
        return a.projPt - b.projPt;
    });
    let splitPt0 = intersections[0];
    let splitPt1 = intersections[intersections.length - 1];
    if (splitPt1.idx < splitPt0.idx) {
        const tmp = splitPt0;
        splitPt0 = splitPt1;
        splitPt1 = tmp;
    }

    const splitPt0Arr = [splitPt0.pt.x, splitPt0.pt.y];
    const splitPt1Arr = [splitPt1.pt.x, splitPt1.pt.y];

    const newPolyA: number[][] = [splitPt0Arr];
    const newPolyB: number[][] = [splitPt1Arr];

    for (let i = splitPt0.idx + 1; i <= splitPt1.idx; i++) {
        addToPoly(newPolyA, points[i].slice());
    }
    addToPoly(newPolyA, splitPt1Arr);
    // Close the path
    addToPoly(newPolyA, splitPt0Arr);

    for (let i = splitPt1.idx + 1; i <= splitPt0.idx + len; i++) {
        addToPoly(newPolyB, points[i % len].slice());
    }
    addToPoly(newPolyB, splitPt0Arr);
    // Close the path
    addToPoly(newPolyB, splitPt1Arr);

    return [{
        points: newPolyA
    }, {
        points: newPolyB
    }];
}

function binaryDividePolygon(
    polygonShape: Pick<Polygon['shape'], 'points'>
) {
    const points = polygonShape.points;
    const min: number[] = [];
    const max: number[] = [];
    fromPoints(points, min, max);
    const boundingRect = new BoundingRect(
        min[0], min[1], max[0] - min[0], max[1] - min[1]
    );

    const width = boundingRect.width;
    const height = boundingRect.height;
    const x = boundingRect.x;
    const y = boundingRect.y;

    const pt0 = new Point();
    const pt1 = new Point();
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


function binaryDivideRecursive<T extends Path['shape']>(
    divider: BinaryDivide, shape: T, count: number, out: T[]
): T[] {
    if (count === 1) {
        out.push(shape);
    }
    else {
        const mid = Math.floor(count / 2);
        const sub = divider(shape);
        binaryDivideRecursive(divider, sub[0], mid, out);
        binaryDivideRecursive(divider, sub[1], count - mid, out);
    }

    return out;
}

export function clone(path: Path, count: number) {
    const paths = [];
    for (let i = 0; i < count; i++) {
        paths.push(clonePath(path));
    }
    return paths;
}

function copyPathProps(source: Path, target: Path) {
    target.setStyle(source.style);
    target.z = source.z;
    target.z2 = source.z2;
    target.zlevel = source.zlevel;
}

function polygonConvert(points: number[]): number[][] {
    const out = [];
    for (let i = 0; i < points.length;) {
        out.push([points[i++], points[i++]]);
    }
    return out;
}

export function split(
    path: Path, count: number
) {
    const outShapes: Path['shape'][] = [];
    const shape = path.shape;
    let OutShapeCtor: new() => Path;
    // TODO Use clone when shape size is small
    switch (path.type) {
        case 'rect':
            divideRect(shape as Rect['shape'], count, outShapes as Rect['shape'][]);
            OutShapeCtor = Rect;
            break;
        case 'sector':
            divideSector(shape as Sector['shape'], count, outShapes as Sector['shape'][]);
            OutShapeCtor = Sector;
            break;
        case 'circle':
            divideSector({
                r0: 0, r: shape.r, startAngle: 0, endAngle: Math.PI * 2,
                cx: shape.cx, cy: shape.cy
            } as Sector['shape'], count, outShapes as Sector['shape'][]);
            OutShapeCtor = Sector;
            break;
        default:
            const m = path.getComputedTransform();
            const scale = m ? Math.sqrt(Math.max(m[0] * m[0] + m[1] * m[1], m[2] * m[2] + m[3] * m[3])) : 1;
            const polygons = map(
                pathToPolygons(path.getUpdatedPathProxy(), scale),
                poly => polygonConvert(poly)
            );
            const polygonCount = polygons.length;
            if (polygonCount === 0) {
                binaryDivideRecursive(binaryDividePolygon, {
                    points: polygons[0]
                }, count, outShapes);
            }
            else if (polygonCount === count) {   // In case we only split batched paths to non-batched paths. No need to split.
                for (let i = 0; i < polygonCount; i++) {
                    outShapes.push({
                        points: polygons[i]
                    } as Polygon['shape']);
                }
            }
            else {
                // Most complex case. Assign multiple subpath to each polygon based on it's area.
                let totalArea = 0;
                const items = map(polygons, poly => {
                    const min: number[] = [];
                    const max: number[] = [];
                    fromPoints(poly, min, max);
                    // TODO: polygon area?
                    const area = (max[1] - min[1]) * (max[0] - min[0]);
                    totalArea += area;
                    return { poly, area };
                });
                items.sort((a, b) => b.area - a.area);

                let left = count;
                for (let i = 0; i < polygonCount; i++) {
                    const item = items[i];
                    if (left <= 0) {
                        break;
                    }

                    const selfCount = i === polygonCount - 1
                        ? left   // Use the last piece directly
                        : Math.ceil(item.area / totalArea * count);

                    if (selfCount < 0) {
                        continue;
                    }

                    binaryDivideRecursive(binaryDividePolygon, {
                        points: item.poly
                    }, selfCount, outShapes);
                    left -= selfCount;
                };
            }
            OutShapeCtor = Polygon;
            break;
    }

    if (!OutShapeCtor) {
        // Unkown split algorithm. Use clone instead
        return clone(path, count);
    }
    const out: Path[] = [];

    for (let i = 0; i < outShapes.length; i++) {
        const subPath = new OutShapeCtor();
        subPath.setShape(outShapes[i]);
        copyPathProps(path, subPath);
        out.push(subPath);
    }

    return out;
}