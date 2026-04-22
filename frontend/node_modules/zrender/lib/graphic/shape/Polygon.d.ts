import Path, { PathProps } from '../Path';
import { VectorArray } from '../../core/vector';
export declare class PolygonShape {
    points: VectorArray[];
    smooth?: number;
    smoothConstraint?: VectorArray[];
}
export interface PolygonProps extends PathProps {
    shape?: Partial<PolygonShape>;
}
declare class Polygon extends Path<PolygonProps> {
    shape: PolygonShape;
    constructor(opts?: PolygonProps);
    getDefaultShape(): PolygonShape;
    buildPath(ctx: CanvasRenderingContext2D, shape: PolygonShape): void;
}
export default Polygon;
