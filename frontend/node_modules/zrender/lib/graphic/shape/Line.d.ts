import Path, { PathProps } from '../Path';
import { VectorArray } from '../../core/vector';
export declare class LineShape {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    percent: number;
}
export interface LineProps extends PathProps {
    shape?: Partial<LineShape>;
}
declare class Line extends Path<LineProps> {
    shape: LineShape;
    constructor(opts?: LineProps);
    getDefaultStyle(): {
        stroke: string;
        fill: string;
    };
    getDefaultShape(): LineShape;
    buildPath(ctx: CanvasRenderingContext2D, shape: LineShape): void;
    pointAt(p: number): VectorArray;
}
export default Line;
