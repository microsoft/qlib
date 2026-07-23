import Path, { PathProps } from '../Path';
export declare class RingShape {
    cx: number;
    cy: number;
    r: number;
    r0: number;
}
export interface RingProps extends PathProps {
    shape?: Partial<RingShape>;
}
declare class Ring extends Path<RingProps> {
    shape: RingShape;
    constructor(opts?: RingProps);
    getDefaultShape(): RingShape;
    buildPath(ctx: CanvasRenderingContext2D, shape: RingShape): void;
}
export default Ring;
