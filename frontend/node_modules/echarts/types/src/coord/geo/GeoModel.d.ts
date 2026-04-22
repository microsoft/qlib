import ComponentModel from '../../model/Component.js';
import Model from '../../model/Model.js';
import Geo from './Geo.js';
import { ComponentOption, BoxLayoutOptionMixin, ItemStyleOption, ZRColor, LabelOption, DisplayState, RoamOptionMixin, AnimationOptionMixin, StatesOptionMixin, Dictionary, CommonTooltipOption, StatesMixinBase } from '../../util/types.js';
import { GeoProjection, NameMap } from './geoTypes.js';
import GlobalModel from '../../model/Global.js';
export interface GeoItemStyleOption<TCbParams = never> extends ItemStyleOption<TCbParams> {
    areaColor?: ZRColor;
}
interface GeoLabelOption extends LabelOption {
    formatter?: string | ((params: GeoLabelFormatterDataParams) => string);
}
export interface GeoStateOption {
    itemStyle?: GeoItemStyleOption;
    label?: GeoLabelOption;
}
interface GeoLabelFormatterDataParams {
    name: string;
    status: DisplayState;
}
export interface RegionOption extends GeoStateOption, StatesOptionMixin<GeoStateOption, StatesMixinBase> {
    name?: string;
    selected?: boolean;
    tooltip?: CommonTooltipOption<GeoTooltipFormatterParams>;
    silent?: boolean;
}
/**
 * @deprecated Use `RegionOption` instead.
 */
export interface RegoinOption extends RegionOption {
}
export interface GeoTooltipFormatterParams {
    componentType: 'geo';
    geoIndex: number;
    name: string;
    $vars: ['name'];
}
export interface GeoCommonOptionMixin extends RoamOptionMixin {
    map: string;
    aspectScale?: number;
    layoutCenter?: (number | string)[];
    layoutSize?: number | string;
    boundingCoords?: number[][];
    nameMap?: NameMap;
    nameProperty?: string;
    /**
     * Use raw projection by default
     * Only available for GeoJSON source.
     *
     * NOTE: `center` needs to be the projected coord if projection is used.
     */
    projection?: GeoProjection;
}
export interface GeoOption extends ComponentOption, BoxLayoutOptionMixin, AnimationOptionMixin, GeoCommonOptionMixin, StatesOptionMixin<GeoStateOption, StatesMixinBase>, GeoStateOption {
    mainType?: 'geo';
    show?: boolean;
    silent?: boolean;
    regions?: RegionOption[];
    stateAnimation?: AnimationOptionMixin;
    selectedMode?: 'single' | 'multiple' | boolean;
    selectedMap?: Dictionary<boolean>;
    tooltip?: CommonTooltipOption<GeoTooltipFormatterParams>;
}
declare class GeoModel extends ComponentModel<GeoOption> {
    static type: string;
    readonly type: string;
    coordinateSystem: Geo;
    static layoutMode: "box";
    private _optionModelMap;
    static defaultOption: GeoOption;
    init(option: GeoOption, parentModel: Model, ecModel: GlobalModel): void;
    optionUpdated(): void;
    /**
     * Get model of region.
     */
    getRegionModel(name: string): Model<RegionOption>;
    /**
     * Format label
     * @param name Region name
     */
    getFormattedLabel(name: string, status?: DisplayState): string;
    setZoom(zoom: number): void;
    setCenter(center: number[]): void;
    select(name?: string): void;
    unSelect(name?: string): void;
    toggleSelected(name?: string): void;
    isSelected(name?: string): boolean;
}
export default GeoModel;
