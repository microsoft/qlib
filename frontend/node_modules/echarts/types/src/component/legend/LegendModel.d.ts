import Model from '../../model/Model.js';
import ComponentModel from '../../model/Component.js';
import { ComponentOption, BoxLayoutOptionMixin, BorderOptionMixin, ColorString, LabelOption, LayoutOrient, CommonTooltipOption, ItemStyleOption, LineStyleOption } from '../../util/types.js';
import { Dictionary } from 'zrender/lib/core/types.js';
import GlobalModel from '../../model/Global.js';
import { ItemStyleProps } from '../../model/mixin/itemStyle.js';
import { LineStyleProps } from './../../model/mixin/lineStyle.js';
import { PathStyleProps } from 'zrender/lib/graphic/Path.js';
declare type SelectorType = 'all' | 'inverse';
export interface LegendSelectorButtonOption {
    type?: SelectorType;
    title?: string;
}
/**
 * T: the type to be extended
 * ET: extended type for keys of T
 * ST: special type for T to be extended
 */
declare type ExtendPropertyType<T, ET, ST extends {
    [key in keyof T]: any;
}> = {
    [key in keyof T]: key extends keyof ST ? T[key] | ET | ST[key] : T[key] | ET;
};
export interface LegendItemStyleOption extends ExtendPropertyType<ItemStyleOption, 'inherit', {
    borderWidth: 'auto';
}> {
}
export interface LegendLineStyleOption extends ExtendPropertyType<LineStyleOption, 'inherit', {
    width: 'auto';
}> {
    inactiveColor?: ColorString;
    inactiveWidth?: number;
}
export interface LegendStyleOption {
    /**
     * Icon of the legend items.
     * @default 'roundRect'
     */
    icon?: string;
    /**
     * Color when legend item is not selected
     */
    inactiveColor?: ColorString;
    /**
     * Border color when legend item is not selected
     */
    inactiveBorderColor?: ColorString;
    /**
     * Border color when legend item is not selected
     */
    inactiveBorderWidth?: number | 'auto';
    /**
     * Legend label formatter
     */
    formatter?: string | ((name: string) => string);
    itemStyle?: LegendItemStyleOption;
    lineStyle?: LegendLineStyleOption;
    textStyle?: LabelOption;
    symbolRotate?: number | 'inherit';
    /**
     * @deprecated
     */
    symbolKeepAspect?: boolean;
}
interface DataItem extends LegendStyleOption {
    name?: string;
    icon?: string;
    textStyle?: LabelOption;
    tooltip?: unknown;
}
export interface LegendTooltipFormatterParams {
    componentType: 'legend';
    legendIndex: number;
    name: string;
    $vars: ['name'];
}
export interface LegendIconParams {
    itemWidth: number;
    itemHeight: number;
    /**
     * symbolType is from legend.icon, legend.data.icon, or series visual
     */
    icon: string;
    iconRotate: number | 'inherit';
    symbolKeepAspect: boolean;
    itemStyle: PathStyleProps;
    lineStyle: LineStyleProps;
}
export interface LegendSymbolStyleOption {
    itemStyle?: ItemStyleProps;
    lineStyle?: LineStyleProps;
}
export interface LegendOption extends ComponentOption, LegendStyleOption, BoxLayoutOptionMixin, BorderOptionMixin {
    mainType?: 'legend';
    show?: boolean;
    orient?: LayoutOrient;
    align?: 'auto' | 'left' | 'right';
    backgroundColor?: ColorString;
    /**
     * Border radius of background rect
     * @default 0
     */
    borderRadius?: number | number[];
    /**
     * Padding between legend item and border.
     * Support to be a single number or an array.
     * @default 5
     */
    padding?: number | number[];
    /**
     * Gap between each legend item.
     * @default 10
     */
    itemGap?: number;
    /**
     * Width of legend symbol
     */
    itemWidth?: number;
    /**
     * Height of legend symbol
     */
    itemHeight?: number;
    selectedMode?: boolean | 'single' | 'multiple';
    /**
     * selected map of each item. Default to be selected if item is not in the map
     */
    selected?: Dictionary<boolean>;
    /**
     * Buttons for all select or inverse select.
     * @example
     *  selector: [{type: 'all or inverse', title: xxx}]
     *  selector: true
     *  selector: ['all', 'inverse']
     */
    selector?: (LegendSelectorButtonOption | SelectorType)[] | boolean;
    selectorLabel?: LabelOption;
    emphasis?: {
        selectorLabel?: LabelOption;
    };
    /**
     * Position of selector buttons.
     */
    selectorPosition?: 'auto' | 'start' | 'end';
    /**
     * Gap between each selector button
     */
    selectorItemGap?: number;
    /**
     * Gap between selector buttons group and legend main items.
     */
    selectorButtonGap?: number;
    data?: (string | DataItem)[];
    /**
     * Tooltip option
     */
    tooltip?: CommonTooltipOption<LegendTooltipFormatterParams>;
}
declare class LegendModel<Ops extends LegendOption = LegendOption> extends ComponentModel<Ops> {
    static type: string;
    type: string;
    static readonly dependencies: string[];
    readonly layoutMode: {
        readonly type: "box";
        readonly ignoreSize: true;
    };
    private _data;
    private _availableNames;
    init(option: Ops, parentModel: Model, ecModel: GlobalModel): void;
    mergeOption(option: Ops, ecModel: GlobalModel): void;
    _updateSelector(option: Ops): void;
    optionUpdated(): void;
    _updateData(ecModel: GlobalModel): void;
    getData(): Model<DataItem>[];
    select(name: string): void;
    unSelect(name: string): void;
    toggleSelected(name: string): void;
    allSelect(): void;
    inverseSelect(): void;
    isSelected(name: string): boolean;
    getOrient(): {
        index: 0;
        name: 'horizontal';
    };
    getOrient(): {
        index: 1;
        name: 'vertical';
    };
    static defaultOption: LegendOption;
}
export default LegendModel;
