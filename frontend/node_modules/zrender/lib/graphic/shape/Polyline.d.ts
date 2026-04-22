import Path, { PathProps } from '../Path';
import { VectorArray } from '../../core/vector';
export declare class PolylineShape {
    points: VectorArray[];
    percent?: number;
    smooth?: number;
    smoothConstraint?: VectorArray[];
}
export interface PolylineProps extends PathProps {
    shape?: Partial<PolylineShape>;
}
declare class Polyline extends Path<PolylineProps> {
    shape: PolylineShape;
    constructor(opts?: PolylineProps);
    getDefaultStyle(): {
        stroke: string;
        fill: string;
    };
    getDefaultShape(): PolylineShape;
    buildPath(ctx: CanvasRenderingContext2D, shape: PolylineShape): void;
}
export default Polyline;
