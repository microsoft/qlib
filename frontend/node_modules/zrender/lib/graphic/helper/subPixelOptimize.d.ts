import { PathStyleProps } from '../Path';
declare type LineShape = {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
};
declare type RectShape = {
    x: number;
    y: number;
    width: number;
    height: number;
    r?: number | number[];
};
export declare function subPixelOptimizeLine(outputShape: Partial<LineShape>, inputShape: LineShape, style: Pick<PathStyleProps, 'lineWidth'>): LineShape;
export declare function subPixelOptimizeRect(outputShape: Partial<RectShape>, inputShape: RectShape, style: Pick<PathStyleProps, 'lineWidth'>): RectShape;
export declare function subPixelOptimize(position: number, lineWidth?: number, positiveOrNegative?: boolean): number;
export {};
