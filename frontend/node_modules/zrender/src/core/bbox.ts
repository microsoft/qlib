/**
 * @author Yi Shen(https://github.com/pissang)
 */

import * as vec2 from './vector';
import * as curve from './curve';

const mathMin = Math.min;
const mathMax = Math.max;
const mathSin = Math.sin;
const mathCos = Math.cos;
const PI2 = Math.PI * 2;

const start = vec2.create();
const end = vec2.create();
const extremity = vec2.create();

/**
 * 从顶点数组中计算出最小包围盒，写入`min`和`max`中
 */
export function fromPoints(points: ArrayLike<number>[], min: vec2.VectorArray, max: vec2.VectorArray) {
    if (points.length === 0) {
        return;
    }
    let p = points[0];
    let left = p[0];
    let right = p[0];
    let top = p[1];
    let bottom = p[1];

    for (let i = 1; i < points.length; i++) {
        p = points[i];
        left = mathMin(left, p[0]);
        right = mathMax(right, p[0]);
        top = mathMin(top, p[1]);
        bottom = mathMax(bottom, p[1]);
    }

    min[0] = left;
    min[1] = top;
    max[0] = right;
    max[1] = bottom;
}

export function fromLine(
    x0: number, y0: number, x1: number, y1: number,
    min: vec2.VectorArray, max: vec2.VectorArray
) {
    min[0] = mathMin(x0, x1);
    min[1] = mathMin(y0, y1);
    max[0] = mathMax(x0, x1);
    max[1] = mathMax(y0, y1);
}

const xDim: number[] = [];
const yDim: number[] = [];
/**
 * 从三阶贝塞尔曲线(p0, p1, p2, p3)中计算出最小包围盒，写入`min`和`max`中
 */
export function fromCubic(
    x0: number, y0: number, x1: number, y1: number, x2: number, y2: number, x3: number, y3: number,
    min: vec2.VectorArray, max: vec2.VectorArray
) {
    const cubicExtrema = curve.cubicExtrema;
    const cubicAt = curve.cubicAt;
    let n = cubicExtrema(x0, x1, x2, x3, xDim);
    min[0] = Infinity;
    min[1] = Infinity;
    max[0] = -Infinity;
    max[1] = -Infinity;

    for (let i = 0; i < n; i++) {
        const x = cubicAt(x0, x1, x2, x3, xDim[i]);
        min[0] = mathMin(x, min[0]);
        max[0] = mathMax(x, max[0]);
    }
    n = cubicExtrema(y0, y1, y2, y3, yDim);
    for (let i = 0; i < n; i++) {
        const y = cubicAt(y0, y1, y2, y3, yDim[i]);
        min[1] = mathMin(y, min[1]);
        max[1] = mathMax(y, max[1]);
    }

    min[0] = mathMin(x0, min[0]);
    max[0] = mathMax(x0, max[0]);
    min[0] = mathMin(x3, min[0]);
    max[0] = mathMax(x3, max[0]);

    min[1] = mathMin(y0, min[1]);
    max[1] = mathMax(y0, max[1]);
    min[1] = mathMin(y3, min[1]);
    max[1] = mathMax(y3, max[1]);
}

/**
 * 从二阶贝塞尔曲线(p0, p1, p2)中计算出最小包围盒，写入`min`和`max`中
 */
export function fromQuadratic(
    x0: number, y0: number, x1: number, y1: number, x2: number, y2: number,
    min: vec2.VectorArray, max: vec2.VectorArray
) {
    const quadraticExtremum = curve.quadraticExtremum;
    const quadraticAt = curve.quadraticAt;
    // Find extremities, where derivative in x dim or y dim is zero
    const tx =
        mathMax(
            mathMin(quadraticExtremum(x0, x1, x2), 1), 0
        );
    const ty =
        mathMax(
            mathMin(quadraticExtremum(y0, y1, y2), 1), 0
        );

    const x = quadraticAt(x0, x1, x2, tx);
    const y = quadraticAt(y0, y1, y2, ty);

    min[0] = mathMin(x0, x2, x);
    min[1] = mathMin(y0, y2, y);
    max[0] = mathMax(x0, x2, x);
    max[1] = mathMax(y0, y2, y);
}

/**
 * 从圆弧中计算出最小包围盒，写入`min`和`max`中
 */
export function fromArc(
    x: number, y: number, rx: number, ry: number, startAngle: number, endAngle: number, anticlockwise: boolean,
    min: vec2.VectorArray, max: vec2.VectorArray
) {
    const vec2Min = vec2.min;
    const vec2Max = vec2.max;

    const diff = Math.abs(startAngle - endAngle);


    if (diff % PI2 < 1e-4 && diff > 1e-4) {
        // Is a circle
        min[0] = x - rx;
        min[1] = y - ry;
        max[0] = x + rx;
        max[1] = y + ry;
        return;
    }

    start[0] = mathCos(startAngle) * rx + x;
    start[1] = mathSin(startAngle) * ry + y;

    end[0] = mathCos(endAngle) * rx + x;
    end[1] = mathSin(endAngle) * ry + y;

    vec2Min(min, start, end);
    vec2Max(max, start, end);

    // Thresh to [0, Math.PI * 2]
    startAngle = startAngle % (PI2);
    if (startAngle < 0) {
        startAngle = startAngle + PI2;
    }
    endAngle = endAngle % (PI2);
    if (endAngle < 0) {
        endAngle = endAngle + PI2;
    }

    if (startAngle > endAngle && !anticlockwise) {
        endAngle += PI2;
    }
    else if (startAngle < endAngle && anticlockwise) {
        startAngle += PI2;
    }
    if (anticlockwise) {
        const tmp = endAngle;
        endAngle = startAngle;
        startAngle = tmp;
    }

    // const number = 0;
    // const step = (anticlockwise ? -Math.PI : Math.PI) / 2;
    for (let angle = 0; angle < endAngle; angle += Math.PI / 2) {
        if (angle > startAngle) {
            extremity[0] = mathCos(angle) * rx + x;
            extremity[1] = mathSin(angle) * ry + y;

            vec2Min(min, extremity, min);
            vec2Max(max, extremity, max);
        }
    }
}
