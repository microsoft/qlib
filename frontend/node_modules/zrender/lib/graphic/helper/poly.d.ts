import { VectorArray } from '../../core/vector';
import PathProxy from '../../core/PathProxy';
export declare function buildPath(ctx: CanvasRenderingContext2D | PathProxy, shape: {
    points: VectorArray[];
    smooth?: number;
    smoothConstraint?: VectorArray[];
}, closePath: boolean): void;
