import BaseAxisPointer, { AxisPointerElementOptions } from './BaseAxisPointer.js';
import { ScaleDataValue, VerticalAlign, CommonAxisPointerOption } from '../../util/types.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import SingleAxisModel from '../../coord/single/AxisModel.js';
import Model from '../../model/Model.js';
declare type AxisPointerModel = Model<CommonAxisPointerOption>;
declare class SingleAxisPointer extends BaseAxisPointer {
    /**
     * @override
     */
    makeElOption(elOption: AxisPointerElementOptions, value: ScaleDataValue, axisModel: SingleAxisModel, axisPointerModel: AxisPointerModel, api: ExtensionAPI): void;
    /**
     * @override
     */
    getHandleTransform(value: ScaleDataValue, axisModel: SingleAxisModel, axisPointerModel: AxisPointerModel): {
        x: number;
        y: number;
        rotation: number;
    };
    /**
     * @override
     */
    updateHandleTransform(transform: {
        x: number;
        y: number;
        rotation: number;
    }, delta: number[], axisModel: SingleAxisModel, axisPointerModel: AxisPointerModel): {
        x: number;
        y: number;
        rotation: number;
        cursorPoint: number[];
        tooltipOption: {
            verticalAlign: VerticalAlign;
        };
    };
}
export default SingleAxisPointer;
