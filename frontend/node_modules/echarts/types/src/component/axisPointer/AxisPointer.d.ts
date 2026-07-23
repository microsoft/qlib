import { AxisBaseModel } from '../../coord/AxisBaseModel.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { CommonAxisPointerOption } from '../../util/types.js';
import Model from '../../model/Model.js';
export interface AxisPointer {
    /**
     * If `show` called, axisPointer must be displayed or remain its original status.
     */
    render(axisModel: AxisBaseModel, axisPointerModel: Model<CommonAxisPointerOption>, api: ExtensionAPI, forceRender?: boolean): void;
    /**
     * If `hide` called, axisPointer must be hidden.
     */
    remove(api: ExtensionAPI): void;
    dispose(api: ExtensionAPI): void;
}
