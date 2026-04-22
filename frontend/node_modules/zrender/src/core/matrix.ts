/**
 * 3x2矩阵操作类
 * @exports zrender/tool/matrix
 */

/* global Float32Array */

import {VectorArray} from './vector';

export type MatrixArray = number[]
/**
 * Create a identity matrix.
 */
export function create(): MatrixArray {
    return [1, 0, 0, 1, 0, 0];
}

/**
 * 设置矩阵为单位矩阵
 */
export function identity(out: MatrixArray): MatrixArray {
    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
    out[3] = 1;
    out[4] = 0;
    out[5] = 0;
    return out;
}

/**
 * 复制矩阵
 */
export function copy(out: MatrixArray, m: MatrixArray): MatrixArray {
    out[0] = m[0];
    out[1] = m[1];
    out[2] = m[2];
    out[3] = m[3];
    out[4] = m[4];
    out[5] = m[5];
    return out;
}

/**
 * 矩阵相乘
 */
export function mul(out: MatrixArray, m1: MatrixArray, m2: MatrixArray): MatrixArray {
    // Consider matrix.mul(m, m2, m);
    // where out is the same as m2.
    // So use temp constiable to escape error.
    const out0 = m1[0] * m2[0] + m1[2] * m2[1];
    const out1 = m1[1] * m2[0] + m1[3] * m2[1];
    const out2 = m1[0] * m2[2] + m1[2] * m2[3];
    const out3 = m1[1] * m2[2] + m1[3] * m2[3];
    const out4 = m1[0] * m2[4] + m1[2] * m2[5] + m1[4];
    const out5 = m1[1] * m2[4] + m1[3] * m2[5] + m1[5];
    out[0] = out0;
    out[1] = out1;
    out[2] = out2;
    out[3] = out3;
    out[4] = out4;
    out[5] = out5;
    return out;
}

/**
 * 平移变换
 */
export function translate(out: MatrixArray, a: MatrixArray, v: VectorArray): MatrixArray {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    out[4] = a[4] + v[0];
    out[5] = a[5] + v[1];
    return out;
}

/**
 * 旋转变换
 */
export function rotate(
    out: MatrixArray,
    a: MatrixArray,
    rad: number,
    pivot: VectorArray = [0, 0]
): MatrixArray {
    const aa = a[0];
    const ac = a[2];
    const atx = a[4];
    const ab = a[1];
    const ad = a[3];
    const aty = a[5];
    const st = Math.sin(rad);
    const ct = Math.cos(rad);

    out[0] = aa * ct + ab * st;
    out[1] = -aa * st + ab * ct;
    out[2] = ac * ct + ad * st;
    out[3] = -ac * st + ct * ad;
    out[4] = ct * (atx - pivot[0]) + st * (aty - pivot[1]) + pivot[0];
    out[5] = ct * (aty - pivot[1]) - st * (atx - pivot[0]) + pivot[1];
    return out;
}

/**
 * 缩放变换
 */
export function scale(out: MatrixArray, a: MatrixArray, v: VectorArray): MatrixArray {
    const vx = v[0];
    const vy = v[1];
    out[0] = a[0] * vx;
    out[1] = a[1] * vy;
    out[2] = a[2] * vx;
    out[3] = a[3] * vy;
    out[4] = a[4] * vx;
    out[5] = a[5] * vy;
    return out;
}

/**
 * 求逆矩阵
 */
export function invert(out: MatrixArray, a: MatrixArray): MatrixArray | null {

    const aa = a[0];
    const ac = a[2];
    const atx = a[4];
    const ab = a[1];
    const ad = a[3];
    const aty = a[5];

    let det = aa * ad - ab * ac;
    if (!det) {
        return null;
    }
    det = 1.0 / det;

    out[0] = ad * det;
    out[1] = -ab * det;
    out[2] = -ac * det;
    out[3] = aa * det;
    out[4] = (ac * aty - ad * atx) * det;
    out[5] = (ab * atx - aa * aty) * det;
    return out;
}

/**
 * Clone a new matrix.
 */
export function clone(a: MatrixArray): MatrixArray {
    const b = create();
    copy(b, a);
    return b;
}
