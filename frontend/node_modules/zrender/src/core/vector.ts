/**
 * @deprecated
 * Use zrender.Point class instead
 */
import { MatrixArray } from './matrix';

/* global Float32Array */

// const ArrayCtor = typeof Float32Array === 'undefined'
//     ? Array
//     : Float32Array;

export type VectorArray = number[]
/**
 * 创建一个向量
 */
export function create(x?: number, y?: number): VectorArray {
    if (x == null) {
        x = 0;
    }
    if (y == null) {
        y = 0;
    }
    return [x, y];
}

/**
 * 复制向量数据
 */
export function copy<T extends VectorArray>(out: T, v: VectorArray): T {
    out[0] = v[0];
    out[1] = v[1];
    return out;
}

/**
 * 克隆一个向量
 */
export function clone(v: VectorArray): VectorArray {
    return [v[0], v[1]];
}

/**
 * 设置向量的两个项
 */
export function set<T extends VectorArray>(out: T, a: number, b: number): T {
    out[0] = a;
    out[1] = b;
    return out;
}

/**
 * 向量相加
 */
export function add<T extends VectorArray>(out: T, v1: VectorArray, v2: VectorArray): T {
    out[0] = v1[0] + v2[0];
    out[1] = v1[1] + v2[1];
    return out;
}

/**
 * 向量缩放后相加
 */
export function scaleAndAdd<T extends VectorArray>(out: T, v1: VectorArray, v2: VectorArray, a: number): T {
    out[0] = v1[0] + v2[0] * a;
    out[1] = v1[1] + v2[1] * a;
    return out;
}

/**
 * 向量相减
 */
export function sub<T extends VectorArray>(out: T, v1: VectorArray, v2: VectorArray): T {
    out[0] = v1[0] - v2[0];
    out[1] = v1[1] - v2[1];
    return out;
}

/**
 * 向量长度
 */
export function len(v: VectorArray): number {
    return Math.sqrt(lenSquare(v));
}
export const length = len;

/**
 * 向量长度平方
 */
export function lenSquare(v: VectorArray): number {
    return v[0] * v[0] + v[1] * v[1];
}
export const lengthSquare = lenSquare;

/**
 * 向量乘法
 */
export function mul<T extends VectorArray>(out: T, v1: VectorArray, v2: VectorArray): T {
    out[0] = v1[0] * v2[0];
    out[1] = v1[1] * v2[1];
    return out;
}

/**
 * 向量除法
 */
export function div<T extends VectorArray>(out: T, v1: VectorArray, v2: VectorArray): T {
    out[0] = v1[0] / v2[0];
    out[1] = v1[1] / v2[1];
    return out;
}

/**
 * 向量点乘
 */
export function dot(v1: VectorArray, v2: VectorArray) {
    return v1[0] * v2[0] + v1[1] * v2[1];
}

/**
 * 向量缩放
 */
export function scale<T extends VectorArray>(out: T, v: VectorArray, s: number): T {
    out[0] = v[0] * s;
    out[1] = v[1] * s;
    return out;
}

/**
 * 向量归一化
 */
export function normalize<T extends VectorArray>(out: T, v: VectorArray): T {
    const d = len(v);
    if (d === 0) {
        out[0] = 0;
        out[1] = 0;
    }
    else {
        out[0] = v[0] / d;
        out[1] = v[1] / d;
    }
    return out;
}

/**
 * 计算向量间距离
 */
export function distance(v1: VectorArray, v2: VectorArray): number {
    return Math.sqrt(
        (v1[0] - v2[0]) * (v1[0] - v2[0])
        + (v1[1] - v2[1]) * (v1[1] - v2[1])
    );
}
export const dist = distance;

/**
 * 向量距离平方
 */
export function distanceSquare(v1: VectorArray, v2: VectorArray): number {
    return (v1[0] - v2[0]) * (v1[0] - v2[0])
        + (v1[1] - v2[1]) * (v1[1] - v2[1]);
}
export const distSquare = distanceSquare;

/**
 * 求负向量
 */
export function negate<T extends VectorArray>(out: T, v: VectorArray): T {
    out[0] = -v[0];
    out[1] = -v[1];
    return out;
}

/**
 * 插值两个点
 */
export function lerp<T extends VectorArray>(out: T, v1: VectorArray, v2: VectorArray, t: number): T {
    out[0] = v1[0] + t * (v2[0] - v1[0]);
    out[1] = v1[1] + t * (v2[1] - v1[1]);
    return out;
}

/**
 * 矩阵左乘向量
 */
export function applyTransform<T extends VectorArray>(out: T, v: VectorArray, m: MatrixArray): T {
    const x = v[0];
    const y = v[1];
    out[0] = m[0] * x + m[2] * y + m[4];
    out[1] = m[1] * x + m[3] * y + m[5];
    return out;
}

/**
 * 求两个向量最小值
 */
export function min<T extends VectorArray>(out: T, v1: VectorArray, v2: VectorArray): T {
    out[0] = Math.min(v1[0], v2[0]);
    out[1] = Math.min(v1[1], v2[1]);
    return out;
}

/**
 * 求两个向量最大值
 */
export function max<T extends VectorArray>(out: T, v1: VectorArray, v2: VectorArray): T {
    out[0] = Math.max(v1[0], v2[0]);
    out[1] = Math.max(v1[1], v2[1]);
    return out;
}
