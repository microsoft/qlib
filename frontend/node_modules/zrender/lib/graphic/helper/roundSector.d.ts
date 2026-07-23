import PathProxy from '../../core/PathProxy';
export declare function buildPath(ctx: CanvasRenderingContext2D | PathProxy, shape: {
    cx: number;
    cy: number;
    startAngle: number;
    endAngle: number;
    clockwise?: boolean;
    r?: number;
    r0?: number;
    cornerRadius?: number | number[];
}): void;
