import Axis from '../../coord/Axis.js';
import { ScaleDataValue, ZRTextAlign, ZRTextVerticalAlign, CommonAxisPointerOption } from '../../util/types.js';
import { VectorArray } from 'zrender/lib/core/vector.js';
import GlobalModel from '../../model/Global.js';
import { AxisPointerElementOptions } from './BaseAxisPointer.js';
import { AxisBaseModel } from '../../coord/AxisBaseModel.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import CartesianAxisModel from '../../coord/cartesian/AxisModel.js';
import Model from '../../model/Model.js';
import { PathStyleProps } from 'zrender/lib/graphic/Path.js';
interface LayoutInfo {
    position: VectorArray;
    rotation: number;
    labelOffset?: number;
    /**
     * 1 | -1
     */
    labelDirection?: number;
    labelMargin?: number;
}
declare type AxisPointerModel = Model<CommonAxisPointerOption>;
export declare function buildElStyle(axisPointerModel: AxisPointerModel): PathStyleProps;
/**
 * @param {Function} labelPos {align, verticalAlign, position}
 */
export declare function buildLabelElOption(elOption: AxisPointerElementOptions, axisModel: AxisBaseModel, axisPointerModel: AxisPointerModel, api: ExtensionAPI, labelPos: {
    align?: ZRTextAlign;
    verticalAlign?: ZRTextVerticalAlign;
    position: number[];
}): void;
export declare function getValueLabel(value: ScaleDataValue, axis: Axis, ecModel: GlobalModel, seriesDataIndices: CommonAxisPointerOption['seriesDataIndices'], opt?: {
    precision?: number | 'auto';
    formatter?: CommonAxisPointerOption['label']['formatter'];
}): string;
export declare function getTransformedPosition(axis: Axis, value: ScaleDataValue, layoutInfo: LayoutInfo): number[];
export declare function buildCartesianSingleLabelElOption(value: ScaleDataValue, elOption: AxisPointerElementOptions, layoutInfo: LayoutInfo, axisModel: CartesianAxisModel, axisPointerModel: AxisPointerModel, api: ExtensionAPI): void;
export declare function makeLineShape(p1: number[], p2: number[], xDimIndex?: number): {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
};
export declare function makeRectShape(xy: number[], wh: number[], xDimIndex?: number): {
    x: number;
    y: number;
    width: number;
    height: number;
};
export declare function makeSectorShape(cx: number, cy: number, r0: number, r: number, startAngle: number, endAngle: number): {
    cx: number;
    cy: number;
    r0: number;
    r: number;
    startAngle: number;
    endAngle: number;
    clockwise: boolean;
};
export {};
