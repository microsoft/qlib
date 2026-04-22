import Path, { PathProps } from '../Path';
export declare class BezierCurveShape {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    cpx1: number;
    cpy1: number;
    cpx2?: number;
    cpy2?: number;
    percent: number;
}
export interface BezierCurveProps extends PathProps {
    shape?: Partial<BezierCurveShape>;
}
declare class BezierCurve extends Path<BezierCurveProps> {
    shape: BezierCurveShape;
    constructor(opts?: BezierCurveProps);
    getDefaultStyle(): {
        stroke: string;
        fill: string;
    };
    getDefaultShape(): BezierCurveShape;
    buildPath(ctx: CanvasRenderingContext2D, shape: BezierCurveShape): void;
    pointAt(t: number): number[];
    tangentAt(t: number): number[];
}
export default BezierCurve;
