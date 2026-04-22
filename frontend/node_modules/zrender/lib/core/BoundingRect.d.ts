import * as matrix from './matrix';
import { PointLike } from './Point';
declare class BoundingRect {
    x: number;
    y: number;
    width: number;
    height: number;
    constructor(x: number, y: number, width: number, height: number);
    union(other: BoundingRect): void;
    applyTransform(m: matrix.MatrixArray): void;
    calculateTransform(b: RectLike): matrix.MatrixArray;
    intersect(b: RectLike, mtv?: PointLike): boolean;
    contain(x: number, y: number): boolean;
    clone(): BoundingRect;
    copy(other: RectLike): void;
    plain(): RectLike;
    isFinite(): boolean;
    isZero(): boolean;
    static create(rect: RectLike): BoundingRect;
    static copy(target: RectLike, source: RectLike): void;
    static applyTransform(target: RectLike, source: RectLike, m: matrix.MatrixArray): void;
}
export declare type RectLike = {
    x: number;
    y: number;
    width: number;
    height: number;
};
export default BoundingRect;
