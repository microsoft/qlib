import { MatrixArray } from './matrix';
export interface PointLike {
    x: number;
    y: number;
}
export default class Point {
    x: number;
    y: number;
    constructor(x?: number, y?: number);
    copy(other: PointLike): this;
    clone(): Point;
    set(x: number, y: number): this;
    equal(other: PointLike): boolean;
    add(other: PointLike): this;
    scale(scalar: number): void;
    scaleAndAdd(other: PointLike, scalar: number): void;
    sub(other: PointLike): this;
    dot(other: PointLike): number;
    len(): number;
    lenSquare(): number;
    normalize(): this;
    distance(other: PointLike): number;
    distanceSquare(other: Point): number;
    negate(): this;
    transform(m: MatrixArray): this;
    toArray(out: number[]): number[];
    fromArray(input: number[]): void;
    static set(p: PointLike, x: number, y: number): void;
    static copy(p: PointLike, p2: PointLike): void;
    static len(p: PointLike): number;
    static lenSquare(p: PointLike): number;
    static dot(p0: PointLike, p1: PointLike): number;
    static add(out: PointLike, p0: PointLike, p1: PointLike): void;
    static sub(out: PointLike, p0: PointLike, p1: PointLike): void;
    static scale(out: PointLike, p0: PointLike, scalar: number): void;
    static scaleAndAdd(out: PointLike, p0: PointLike, p1: PointLike, scalar: number): void;
    static lerp(out: PointLike, p0: PointLike, p1: PointLike, t: number): void;
}
