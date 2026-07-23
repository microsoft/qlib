import Path, { PathProps } from '../Path';
export declare class TrochoidShape {
    cx: number;
    cy: number;
    r: number;
    r0: number;
    d: number;
    location: string;
}
export interface TrochoidProps extends PathProps {
    shape?: Partial<TrochoidShape>;
}
declare class Trochoid extends Path<TrochoidProps> {
    shape: TrochoidShape;
    constructor(opts?: TrochoidProps);
    getDefaultStyle(): {
        stroke: string;
        fill: string;
    };
    getDefaultShape(): TrochoidShape;
    buildPath(ctx: CanvasRenderingContext2D, shape: TrochoidShape): void;
}
export default Trochoid;
