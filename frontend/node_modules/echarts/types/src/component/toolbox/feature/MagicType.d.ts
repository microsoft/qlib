import { ToolboxFeature, ToolboxFeatureOption } from '../featureManager.js';
import { SeriesOption } from '../../../util/types.js';
import GlobalModel from '../../../model/Global.js';
import ExtensionAPI from '../../../core/ExtensionAPI.js';
declare const ICON_TYPES: readonly ["line", "bar", "stack"];
declare const TITLE_TYPES: readonly ["line", "bar", "stack", "tiled"];
declare type IconType = typeof ICON_TYPES[number];
declare type TitleType = typeof TITLE_TYPES[number];
export interface ToolboxMagicTypeFeatureOption extends ToolboxFeatureOption {
    type?: IconType[];
    /**
     * Icon group
     */
    icon?: {
        [key in IconType]?: string;
    };
    title?: {
        [key in TitleType]?: string;
    };
    option?: {
        [key in IconType]?: SeriesOption;
    };
    /**
     * Map of seriesType: seriesIndex
     */
    seriesIndex?: {
        line?: number;
        bar?: number;
    };
}
declare class MagicType extends ToolboxFeature<ToolboxMagicTypeFeatureOption> {
    getIcons(): {
        line?: string;
        stack?: string;
        bar?: string;
    };
    static getDefaultOption(ecModel: GlobalModel): ToolboxMagicTypeFeatureOption;
    onclick(ecModel: GlobalModel, api: ExtensionAPI, type: IconType): void;
}
export default MagicType;
