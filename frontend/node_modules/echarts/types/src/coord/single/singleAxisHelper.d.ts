import SingleAxisModel from './AxisModel.js';
interface LayoutResult {
    position: [number, number];
    rotation: number;
    labelRotation: number;
    labelDirection: number;
    tickDirection: number;
    nameDirection: number;
    z2: number;
}
export declare function layout(axisModel: SingleAxisModel, opt?: {
    labelInside?: boolean;
    rotate?: number;
}): LayoutResult;
export {};
