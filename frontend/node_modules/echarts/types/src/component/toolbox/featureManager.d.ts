import { Dictionary, DisplayState, ZRElementEvent, ItemStyleOption, LabelOption } from '../../util/types.js';
import Model from '../../model/Model.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import Displayable from 'zrender/lib/graphic/Displayable.js';
declare type IconStyle = ItemStyleOption & {
    textFill?: LabelOption['color'];
    textBackgroundColor?: LabelOption['backgroundColor'];
    textPosition?: LabelOption['position'];
    textAlign?: LabelOption['align'];
    textBorderRadius?: LabelOption['borderRadius'];
    textPadding?: LabelOption['padding'];
    textFontFamily?: LabelOption['fontFamily'];
    textFontSize?: LabelOption['fontSize'];
    textFontWeight?: LabelOption['fontWeight'];
    textFontStyle?: LabelOption['fontStyle'];
};
export interface ToolboxFeatureOption {
    show?: boolean;
    title?: string | Partial<Dictionary<string>>;
    icon?: string | Partial<Dictionary<string>>;
    iconStyle?: IconStyle;
    emphasis?: {
        iconStyle?: IconStyle;
    };
    iconStatus?: Partial<Dictionary<DisplayState>>;
    onclick?: () => void;
}
export interface ToolboxFeatureModel<Opts extends ToolboxFeatureOption = ToolboxFeatureOption> extends Model<Opts> {
    /**
     * Collection of icon paths.
     * Will be injected during rendering in the view.
     */
    iconPaths: Partial<Dictionary<Displayable>>;
    setIconStatus(iconName: string, status: DisplayState): void;
}
interface ToolboxFeature<Opts extends ToolboxFeatureOption = ToolboxFeatureOption> {
    getIcons?(): Dictionary<string>;
    onclick(ecModel: GlobalModel, api: ExtensionAPI, type: string, event: ZRElementEvent): void;
    dispose?(ecModel: GlobalModel, api: ExtensionAPI): void;
    remove?(ecModel: GlobalModel, api: ExtensionAPI): void;
    render(featureModel: ToolboxFeatureModel, model: GlobalModel, api: ExtensionAPI, payload: unknown): void;
    updateView?(featureModel: ToolboxFeatureModel, model: GlobalModel, api: ExtensionAPI, payload: unknown): void;
}
declare abstract class ToolboxFeature<Opts extends ToolboxFeatureOption = ToolboxFeatureOption> {
    uid: string;
    model: ToolboxFeatureModel<Opts>;
    ecModel: GlobalModel;
    api: ExtensionAPI;
    /**
     * If toolbox feature can't be used on some platform.
     */
    unusable?: boolean;
}
export { ToolboxFeature };
export interface UserDefinedToolboxFeature {
    uid: string;
    model: ToolboxFeatureModel;
    ecModel: GlobalModel;
    api: ExtensionAPI;
    featureName?: string;
    onclick(): void;
}
declare type ToolboxFeatureCtor = {
    new (): ToolboxFeature;
    /**
     * Static defaultOption property
     */
    defaultOption?: ToolboxFeatureOption;
    getDefaultOption?: (ecModel: GlobalModel) => ToolboxFeatureOption;
};
export declare function registerFeature(name: string, ctor: ToolboxFeatureCtor): void;
export declare function getFeature(name: string): ToolboxFeatureCtor;
