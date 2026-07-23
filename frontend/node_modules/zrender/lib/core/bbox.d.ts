import * as vec2 from './vector';
export declare function fromPoints(points: ArrayLike<number>[], min: vec2.VectorArray, max: vec2.VectorArray): void;
export declare function fromLine(x0: number, y0: number, x1: number, y1: number, min: vec2.VectorArray, max: vec2.VectorArray): void;
export declare function fromCubic(x0: number, y0: number, x1: number, y1: number, x2: number, y2: number, x3: number, y3: number, min: vec2.VectorArray, max: vec2.VectorArray): void;
export declare function fromQuadratic(x0: number, y0: number, x1: number, y1: number, x2: number, y2: number, min: vec2.VectorArray, max: vec2.VectorArray): void;
export declare function fromArc(x: number, y: number, rx: number, ry: number, startAngle: number, endAngle: number, anticlockwise: boolean, min: vec2.VectorArray, max: vec2.VectorArray): void;
