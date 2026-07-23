import { ToolboxFeature, ToolboxFeatureOption } from '../featureManager.js';
import { ZRColor } from '../../../util/types.js';
import GlobalModel from '../../../model/Global.js';
import ExtensionAPI from '../../../core/ExtensionAPI.js';
export interface ToolboxSaveAsImageFeatureOption extends ToolboxFeatureOption {
    icon?: string;
    title?: string;
    type?: 'png' | 'jpeg';
    backgroundColor?: ZRColor;
    connectedBackgroundColor?: ZRColor;
    name?: string;
    excludeComponents?: string[];
    pixelRatio?: number;
    lang?: string[];
}
declare class SaveAsImage extends ToolboxFeature<ToolboxSaveAsImageFeatureOption> {
    onclick(ecModel: GlobalModel, api: ExtensionAPI): void;
    static getDefaultOption(ecModel: GlobalModel): ToolboxSaveAsImageFeatureOption;
}
export default SaveAsImage;
