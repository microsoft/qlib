import Path, { PathProps } from '../Path';
export declare class HeartShape {
    cx: number;
    cy: number;
    width: number;
    height: number;
}
export interface HeartProps extends PathProps {
    shape?: Partial<HeartShape>;
}
declare class Heart extends Path<HeartProps> {
    shape: HeartShape;
    constructor(opts?: HeartProps);
    getDefaultShape(): HeartShape;
    buildPath(ctx: CanvasRenderingContext2D, shape: HeartShape): void;
}
export default Heart;
