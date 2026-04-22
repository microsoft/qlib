import ComponentView from '../../view/Component.js';
import ToolboxModel from './ToolboxModel.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { Dictionary, Payload } from '../../util/types.js';
import { ToolboxFeature, ToolboxFeatureOption, UserDefinedToolboxFeature } from './featureManager.js';
declare class ToolboxView extends ComponentView {
    static type: "toolbox";
    _features: Dictionary<ToolboxFeature | UserDefinedToolboxFeature>;
    _featureNames: string[];
    render(toolboxModel: ToolboxModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload & {
        newTitle?: ToolboxFeatureOption['title'];
    }): void;
    updateView(toolboxModel: ToolboxModel, ecModel: GlobalModel, api: ExtensionAPI, payload: unknown): void;
    remove(ecModel: GlobalModel, api: ExtensionAPI): void;
    dispose(ecModel: GlobalModel, api: ExtensionAPI): void;
}
export default ToolboxView;
