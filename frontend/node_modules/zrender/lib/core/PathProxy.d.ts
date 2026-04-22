import BoundingRect from './BoundingRect';
interface ExtendedCanvasRenderingContext2D extends CanvasRenderingContext2D {
    dpr?: number;
}
export declare function normalizeArcAngles(angles: number[], anticlockwise: boolean): void;
export default class PathProxy {
    dpr: number;
    data: number[] | Float32Array;
    private _version;
    private _saveData;
    private _pendingPtX;
    private _pendingPtY;
    private _pendingPtDist;
    private _ctx;
    private _xi;
    private _yi;
    private _x0;
    private _y0;
    private _len;
    private _pathSegLen;
    private _pathLen;
    private _ux;
    private _uy;
    static CMD: {
        M: number;
        L: number;
        C: number;
        Q: number;
        A: number;
        Z: number;
        R: number;
    };
    constructor(notSaveData?: boolean);
    increaseVersion(): void;
    getVersion(): number;
    setScale(sx: number, sy: number, segmentIgnoreThreshold?: number): void;
    setDPR(dpr: number): void;
    setContext(ctx: ExtendedCanvasRenderingContext2D): void;
    getContext(): ExtendedCanvasRenderingContext2D;
    beginPath(): this;
    reset(): void;
    moveTo(x: number, y: number): this;
    lineTo(x: number, y: number): this;
    bezierCurveTo(x1: number, y1: number, x2: number, y2: number, x3: number, y3: number): this;
    quadraticCurveTo(x1: number, y1: number, x2: number, y2: number): this;
    arc(cx: number, cy: number, r: number, startAngle: number, endAngle: number, anticlockwise?: boolean): this;
    arcTo(x1: number, y1: number, x2: number, y2: number, radius: number): this;
    rect(x: number, y: number, w: number, h: number): this;
    closePath(): this;
    fill(ctx: CanvasRenderingContext2D): void;
    stroke(ctx: CanvasRenderingContext2D): void;
    len(): number;
    setData(data: Float32Array | number[]): void;
    appendPath(path: PathProxy | PathProxy[]): void;
    addData(cmd: number, a?: number, b?: number, c?: number, d?: number, e?: number, f?: number, g?: number, h?: number): void;
    private _drawPendingPt;
    private _expandData;
    toStatic(): void;
    getBoundingRect(): BoundingRect;
    private _calculateLength;
    rebuildPath(ctx: PathRebuilder, percent: number): void;
    clone(): PathProxy;
    private static initDefaultProps;
}
export interface PathRebuilder {
    moveTo(x: number, y: number): void;
    lineTo(x: number, y: number): void;
    bezierCurveTo(x: number, y: number, x2: number, y2: number, x3: number, y3: number): void;
    quadraticCurveTo(x: number, y: number, x2: number, y2: number): void;
    arc(cx: number, cy: number, r: number, startAngle: number, endAngle: number, anticlockwise: boolean): void;
    ellipse(cx: number, cy: number, radiusX: number, radiusY: number, rotation: number, startAngle: number, endAngle: number, anticlockwise: boolean): void;
    rect(x: number, y: number, width: number, height: number): void;
    closePath(): void;
}
export {};
