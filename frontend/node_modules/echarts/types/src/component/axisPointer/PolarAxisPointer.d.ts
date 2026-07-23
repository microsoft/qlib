import BaseAxisPointer, { AxisPointerElementOptions } from './BaseAxisPointer.js';
import { OptionDataValue, CommonAxisPointerOption } from '../../util/types.js';
import { PolarAxisModel } from '../../coord/polar/AxisModel.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import Model from '../../model/Model.js';
declare class PolarAxisPointer extends BaseAxisPointer {
    /**
     * @override
     */
    makeElOption(elOption: AxisPointerElementOptions, value: OptionDataValue, axisModel: PolarAxisModel, axisPointerModel: Model<CommonAxisPointerOption>, api: ExtensionAPI): void;
}
export default PolarAxisPointer;
