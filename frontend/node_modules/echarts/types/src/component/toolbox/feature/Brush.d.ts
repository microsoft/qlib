import { ToolboxFeatureModel, ToolboxFeatureOption, ToolboxFeature } from '../featureManager.js';
import GlobalModel from '../../../model/Global.js';
import ExtensionAPI from '../../../core/ExtensionAPI.js';
declare const ICON_TYPES: readonly ["rect", "polygon", "lineX", "lineY", "keep", "clear"];
declare type IconType = typeof ICON_TYPES[number];
export interface ToolboxBrushFeatureOption extends ToolboxFeatureOption {
    type?: IconType[];
    icon?: {
        [key in IconType]?: string;
    };
    title?: {
        [key in IconType]?: string;
    };
}
declare class BrushFeature extends ToolboxFeature<ToolboxBrushFeatureOption> {
    private _brushType;
    private _brushMode;
    render(featureModel: ToolboxFeatureModel<ToolboxBrushFeatureOption>, ecModel: GlobalModel, api: ExtensionAPI): void;
    updateView(featureModel: ToolboxFeatureModel<ToolboxBrushFeatureOption>, ecModel: GlobalModel, api: ExtensionAPI): void;
    getIcons(): {
        clear?: string;
        polygon?: string;
        rect?: string;
        lineX?: string;
        lineY?: string;
        keep?: string;
    };
    onclick(ecModel: GlobalModel, api: ExtensionAPI, type: IconType): void;
    static getDefaultOption(ecModel: GlobalModel): ToolboxBrushFeatureOption;
}
export default BrushFeature;
