/**
 * 曲线辅助模块
 */

import {
    create as v2Create,
    distSquare as v2DistSquare,
    VectorArray
} from './vector';

const mathPow = Math.pow;
const mathSqrt = Math.sqrt;

const EPSILON = 1e-8;
const EPSILON_NUMERIC = 1e-4;

const THREE_SQRT = mathSqrt(3);
const ONE_THIRD = 1 / 3;

// 临时变量
const _v0 = v2Create();
const _v1 = v2Create();
const _v2 = v2Create();

function isAroundZero(val: number) {
    return val > -EPSILON && val < EPSILON;
}
function isNotAroundZero(val: number) {
    return val > EPSILON || val < -EPSILON;
}
/**
 * 计算三次贝塞尔值
 */
export function cubicAt(p0: number, p1: number, p2: number, p3: number, t: number): number {
    const onet = 1 - t;
    return onet * onet * (onet * p0 + 3 * t * p1)
            + t * t * (t * p3 + 3 * onet * p2);
}

/**
 * 计算三次贝塞尔导数值
 */
export function cubicDerivativeAt(p0: number, p1: number, p2: number, p3: number, t: number): number {
    const onet = 1 - t;
    return 3 * (
        ((p1 - p0) * onet + 2 * (p2 - p1) * t) * onet
        + (p3 - p2) * t * t
    );
}

/**
 * 计算三次贝塞尔方程根，使用盛金公式
 */
export function cubicRootAt(p0: number, p1: number, p2: number, p3: number, val: number, roots: number[]): number {
    // Evaluate roots of cubic functions
    const a = p3 + 3 * (p1 - p2) - p0;
    const b = 3 * (p2 - p1 * 2 + p0);
    const c = 3 * (p1 - p0);
    const d = p0 - val;

    const A = b * b - 3 * a * c;
    const B = b * c - 9 * a * d;
    const C = c * c - 3 * b * d;

    let n = 0;

    if (isAroundZero(A) && isAroundZero(B)) {
        if (isAroundZero(b)) {
            roots[0] = 0;
        }
        else {
            const t1 = -c / b;  //t1, t2, t3, b is not zero
            if (t1 >= 0 && t1 <= 1) {
                roots[n++] = t1;
            }
        }
    }
    else {
        const disc = B * B - 4 * A * C;

        if (isAroundZero(disc)) {
            const K = B / A;
            const t1 = -b / a + K;  // t1, a is not zero
            const t2 = -K / 2;  // t2, t3
            if (t1 >= 0 && t1 <= 1) {
                roots[n++] = t1;
            }
            if (t2 >= 0 && t2 <= 1) {
                roots[n++] = t2;
            }
        }
        else if (disc > 0) {
            const discSqrt = mathSqrt(disc);
            let Y1 = A * b + 1.5 * a * (-B + discSqrt);
            let Y2 = A * b + 1.5 * a * (-B - discSqrt);
            if (Y1 < 0) {
                Y1 = -mathPow(-Y1, ONE_THIRD);
            }
            else {
                Y1 = mathPow(Y1, ONE_THIRD);
            }
            if (Y2 < 0) {
                Y2 = -mathPow(-Y2, ONE_THIRD);
            }
            else {
                Y2 = mathPow(Y2, ONE_THIRD);
            }
            const t1 = (-b - (Y1 + Y2)) / (3 * a);
            if (t1 >= 0 && t1 <= 1) {
                roots[n++] = t1;
            }
        }
        else {
            const T = (2 * A * b - 3 * a * B) / (2 * mathSqrt(A * A * A));
            const theta = Math.acos(T) / 3;
            const ASqrt = mathSqrt(A);
            const tmp = Math.cos(theta);

            const t1 = (-b - 2 * ASqrt * tmp) / (3 * a);
            const t2 = (-b + ASqrt * (tmp + THREE_SQRT * Math.sin(theta))) / (3 * a);
            const t3 = (-b + ASqrt * (tmp - THREE_SQRT * Math.sin(theta))) / (3 * a);
            if (t1 >= 0 && t1 <= 1) {
                roots[n++] = t1;
            }
            if (t2 >= 0 && t2 <= 1) {
                roots[n++] = t2;
            }
            if (t3 >= 0 && t3 <= 1) {
                roots[n++] = t3;
            }
        }
    }
    return n;
}

/**
 * 计算三次贝塞尔方程极限值的位置
 * @return 有效数目
 */
export function cubicExtrema(p0: number, p1: number, p2: number, p3: number, extrema: number[]): number {
    const b = 6 * p2 - 12 * p1 + 6 * p0;
    const a = 9 * p1 + 3 * p3 - 3 * p0 - 9 * p2;
    const c = 3 * p1 - 3 * p0;

    let n = 0;
    if (isAroundZero(a)) {
        if (isNotAroundZero(b)) {
            const t1 = -c / b;
            if (t1 >= 0 && t1 <= 1) {
                extrema[n++] = t1;
            }
        }
    }
    else {
        const disc = b * b - 4 * a * c;
        if (isAroundZero(disc)) {
            extrema[0] = -b / (2 * a);
        }
        else if (disc > 0) {
            const discSqrt = mathSqrt(disc);
            const t1 = (-b + discSqrt) / (2 * a);
            const t2 = (-b - discSqrt) / (2 * a);
            if (t1 >= 0 && t1 <= 1) {
                extrema[n++] = t1;
            }
            if (t2 >= 0 && t2 <= 1) {
                extrema[n++] = t2;
            }
        }
    }
    return n;
}

/**
 * 细分三次贝塞尔曲线
 */
export function cubicSubdivide(p0: number, p1: number, p2: number, p3: number, t: number, out: number[]) {
    const p01 = (p1 - p0) * t + p0;
    const p12 = (p2 - p1) * t + p1;
    const p23 = (p3 - p2) * t + p2;

    const p012 = (p12 - p01) * t + p01;
    const p123 = (p23 - p12) * t + p12;

    const p0123 = (p123 - p012) * t + p012;
    // Seg0
    out[0] = p0;
    out[1] = p01;
    out[2] = p012;
    out[3] = p0123;
    // Seg1
    out[4] = p0123;
    out[5] = p123;
    out[6] = p23;
    out[7] = p3;
}

/**
 * 投射点到三次贝塞尔曲线上，返回投射距离。
 * 投射点有可能会有一个或者多个，这里只返回其中距离最短的一个。
 */
export function cubicProjectPoint(
    x0: number, y0: number, x1: number, y1: number, x2: number, y2: number, x3: number, y3: number,
    x: number, y: number, out: VectorArray
): number {
    // http://pomax.github.io/bezierinfo/#projections
    let t;
    let interval = 0.005;
    let d = Infinity;
    let prev;
    let next;
    let d1;
    let d2;

    _v0[0] = x;
    _v0[1] = y;

    // 先粗略估计一下可能的最小距离的 t 值
    // PENDING
    for (let _t = 0; _t < 1; _t += 0.05) {
        _v1[0] = cubicAt(x0, x1, x2, x3, _t);
        _v1[1] = cubicAt(y0, y1, y2, y3, _t);
        d1 = v2DistSquare(_v0, _v1);
        if (d1 < d) {
            t = _t;
            d = d1;
        }
    }
    d = Infinity;

    // At most 32 iteration
    for (let i = 0; i < 32; i++) {
        if (interval < EPSILON_NUMERIC) {
            break;
        }
        prev = t - interval;
        next = t + interval;
        // t - interval
        _v1[0] = cubicAt(x0, x1, x2, x3, prev);
        _v1[1] = cubicAt(y0, y1, y2, y3, prev);

        d1 = v2DistSquare(_v1, _v0);

        if (prev >= 0 && d1 < d) {
            t = prev;
            d = d1;
        }
        else {
            // t + interval
            _v2[0] = cubicAt(x0, x1, x2, x3, next);
            _v2[1] = cubicAt(y0, y1, y2, y3, next);
            d2 = v2DistSquare(_v2, _v0);

            if (next <= 1 && d2 < d) {
                t = next;
                d = d2;
            }
            else {
                interval *= 0.5;
            }
        }
    }
    // t
    if (out) {
        out[0] = cubicAt(x0, x1, x2, x3, t);
        out[1] = cubicAt(y0, y1, y2, y3, t);
    }
    // console.log(interval, i);
    return mathSqrt(d);
}

/**
 * 计算三次贝塞尔曲线长度
 */
export function cubicLength(
    x0: number, y0: number, x1: number, y1: number, x2: number, y2: number, x3: number, y3: number,
    iteration: number
) {
    let px = x0;
    let py = y0;

    let d = 0;

    const step = 1 / iteration;

    for (let i = 1; i <= iteration; i++) {
        let t = i * step;
        const x = cubicAt(x0, x1, x2, x3, t);
        const y = cubicAt(y0, y1, y2, y3, t);

        const dx = x - px;
        const dy = y - py;

        d += Math.sqrt(dx * dx + dy * dy);

        px = x;
        py = y;
    }

    return d;
}

/**
 * 计算二次方贝塞尔值
 */
export function quadraticAt(p0: number, p1: number, p2: number, t: number): number {
    const onet = 1 - t;
    return onet * (onet * p0 + 2 * t * p1) + t * t * p2;
}

/**
 * 计算二次方贝塞尔导数值
 */
export function quadraticDerivativeAt(p0: number, p1: number, p2: number, t: number): number {
    return 2 * ((1 - t) * (p1 - p0) + t * (p2 - p1));
}

/**
 * 计算二次方贝塞尔方程根
 * @return 有效根数目
 */
export function quadraticRootAt(p0: number, p1: number, p2: number, val: number, roots: number[]): number {
    const a = p0 - 2 * p1 + p2;
    const b = 2 * (p1 - p0);
    const c = p0 - val;

    let n = 0;
    if (isAroundZero(a)) {
        if (isNotAroundZero(b)) {
            const t1 = -c / b;
            if (t1 >= 0 && t1 <= 1) {
                roots[n++] = t1;
            }
        }
    }
    else {
        const disc = b * b - 4 * a * c;
        if (isAroundZero(disc)) {
            const t1 = -b / (2 * a);
            if (t1 >= 0 && t1 <= 1) {
                roots[n++] = t1;
            }
        }
        else if (disc > 0) {
            const discSqrt = mathSqrt(disc);
            const t1 = (-b + discSqrt) / (2 * a);
            const t2 = (-b - discSqrt) / (2 * a);
            if (t1 >= 0 && t1 <= 1) {
                roots[n++] = t1;
            }
            if (t2 >= 0 && t2 <= 1) {
                roots[n++] = t2;
            }
        }
    }
    return n;
}

/**
 * 计算二次贝塞尔方程极限值
 */
export function quadraticExtremum(p0: number, p1: number, p2: number): number {
    const divider = p0 + p2 - 2 * p1;
    if (divider === 0) {
        // p1 is center of p0 and p2
        return 0.5;
    }
    else {
        return (p0 - p1) / divider;
    }
}

/**
 * 细分二次贝塞尔曲线
 */
export function quadraticSubdivide(p0: number, p1: number, p2: number, t: number, out: number[]) {
    const p01 = (p1 - p0) * t + p0;
    const p12 = (p2 - p1) * t + p1;
    const p012 = (p12 - p01) * t + p01;

    // Seg0
    out[0] = p0;
    out[1] = p01;
    out[2] = p012;

    // Seg1
    out[3] = p012;
    out[4] = p12;
    out[5] = p2;
}

/**
 * 投射点到二次贝塞尔曲线上，返回投射距离。
 * 投射点有可能会有一个或者多个，这里只返回其中距离最短的一个。
 * @param {number} x0
 * @param {number} y0
 * @param {number} x1
 * @param {number} y1
 * @param {number} x2
 * @param {number} y2
 * @param {number} x
 * @param {number} y
 * @param {Array.<number>} out 投射点
 * @return {number}
 */
export function quadraticProjectPoint(
    x0: number, y0: number, x1: number, y1: number, x2: number, y2: number,
    x: number, y: number, out: VectorArray
): number {
    // http://pomax.github.io/bezierinfo/#projections
    let t: number;
    let interval = 0.005;
    let d = Infinity;

    _v0[0] = x;
    _v0[1] = y;

    // 先粗略估计一下可能的最小距离的 t 值
    // PENDING
    for (let _t = 0; _t < 1; _t += 0.05) {
        _v1[0] = quadraticAt(x0, x1, x2, _t);
        _v1[1] = quadraticAt(y0, y1, y2, _t);
        const d1 = v2DistSquare(_v0, _v1);
        if (d1 < d) {
            t = _t;
            d = d1;
        }
    }
    d = Infinity;

    // At most 32 iteration
    for (let i = 0; i < 32; i++) {
        if (interval < EPSILON_NUMERIC) {
            break;
        }
        const prev = t - interval;
        const next = t + interval;
        // t - interval
        _v1[0] = quadraticAt(x0, x1, x2, prev);
        _v1[1] = quadraticAt(y0, y1, y2, prev);

        const d1 = v2DistSquare(_v1, _v0);

        if (prev >= 0 && d1 < d) {
            t = prev;
            d = d1;
        }
        else {
            // t + interval
            _v2[0] = quadraticAt(x0, x1, x2, next);
            _v2[1] = quadraticAt(y0, y1, y2, next);
            const d2 = v2DistSquare(_v2, _v0);
            if (next <= 1 && d2 < d) {
                t = next;
                d = d2;
            }
            else {
                interval *= 0.5;
            }
        }
    }
    // t
    if (out) {
        out[0] = quadraticAt(x0, x1, x2, t);
        out[1] = quadraticAt(y0, y1, y2, t);
    }
    // console.log(interval, i);
    return mathSqrt(d);
}

/**
 * 计算二次贝塞尔曲线长度
 */
export function quadraticLength(
    x0: number, y0: number, x1: number, y1: number, x2: number, y2: number,
    iteration: number
) {
    let px = x0;
    let py = y0;

    let d = 0;

    const step = 1 / iteration;

    for (let i = 1; i <= iteration; i++) {
        let t = i * step;
        const x = quadraticAt(x0, x1, x2, t);
        const y = quadraticAt(y0, y1, y2, t);

        const dx = x - px;
        const dy = y - py;

        d += Math.sqrt(dx * dx + dy * dy);

        px = x;
        py = y;
    }

    return d;
}
