import ComponentView from '../../view/Component.js';
import AxisPointerModel from './AxisPointerModel.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
declare class AxisPointerView extends ComponentView {
    static type: "axisPointer";
    type: "axisPointer";
    render(globalAxisPointerModel: AxisPointerModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    remove(ecModel: GlobalModel, api: ExtensionAPI): void;
    dispose(ecModel: GlobalModel, api: ExtensionAPI): void;
}
export default AxisPointerView;
