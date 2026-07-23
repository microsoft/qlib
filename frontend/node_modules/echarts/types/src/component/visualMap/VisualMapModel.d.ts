import VisualMapping, { VisualMappingOption } from '../../visual/VisualMapping.js';
import { ComponentOption, BoxLayoutOptionMixin, LabelOption, ColorString, ZRColor, BorderOptionMixin, OptionDataValue, BuiltinVisualProperty, DimensionIndex } from '../../util/types.js';
import ComponentModel from '../../model/Component.js';
import Model from '../../model/Model.js';
import GlobalModel from '../../model/Global.js';
import SeriesModel from '../../model/Series.js';
import SeriesData from '../../data/SeriesData.js';
declare type VisualOptionBase = {
    [key in BuiltinVisualProperty]?: any;
};
declare type LabelFormatter = (min: OptionDataValue, max?: OptionDataValue) => string;
declare type VisualState = VisualMapModel['stateList'][number];
export interface VisualMapOption<T extends VisualOptionBase = VisualOptionBase> extends ComponentOption, BoxLayoutOptionMixin, BorderOptionMixin {
    mainType?: 'visualMap';
    show?: boolean;
    align?: string;
    realtime?: boolean;
    /**
     * 'all' or null/undefined: all series.
     * A number or an array of number: the specified series.
     * set min: 0, max: 200, only for campatible with ec2.
     * In fact min max should not have default value.
     */
    seriesIndex?: 'all' | number[] | number;
    /**
     * min value, must specified if pieces is not specified.
     */
    min?: number;
    /**
     * max value, must specified if pieces is not specified.
     */
    max?: number;
    /**
     * Dimension to be encoded
     */
    dimension?: number;
    /**
     * Visual configuration for the data in selection
     */
    inRange?: T;
    /**
     * Visual configuration for the out of selection
     */
    outOfRange?: T;
    controller?: {
        inRange?: T;
        outOfRange?: T;
    };
    target?: {
        inRange?: T;
        outOfRange?: T;
    };
    /**
     * Width of the display item
     */
    itemWidth?: number;
    /**
     * Height of the display item
     */
    itemHeight?: number;
    inverse?: boolean;
    orient?: 'horizontal' | 'vertical';
    backgroundColor?: ZRColor;
    contentColor?: ZRColor;
    inactiveColor?: ZRColor;
    /**
     * Padding of the component. Can be an array similar to CSS
     */
    padding?: number[] | number;
    /**
     * Gap between text and item
     */
    textGap?: number;
    precision?: number;
    /**
     * @deprecated
     * Option from version 2
     */
    color?: ColorString[];
    formatter?: string | LabelFormatter;
    /**
     * Text on the both end. Such as ['High', 'Low']
     */
    text?: string[];
    textStyle?: LabelOption;
    categories?: unknown;
}
export interface VisualMeta {
    stops: {
        value: number;
        color: ColorString;
    }[];
    outerColors: ColorString[];
    dimension?: DimensionIndex;
}
declare class VisualMapModel<Opts extends VisualMapOption = VisualMapOption> extends ComponentModel<Opts> {
    static type: string;
    type: string;
    static readonly dependencies: string[];
    readonly stateList: readonly ["inRange", "outOfRange"];
    readonly replacableOptionKeys: readonly ["inRange", "outOfRange", "target", "controller", "color"];
    readonly layoutMode: {
        readonly type: "box";
        readonly ignoreSize: true;
    };
    /**
     * [lowerBound, upperBound]
     */
    dataBound: number[];
    protected _dataExtent: [number, number];
    targetVisuals: {
        [x: string]: {
            symbol?: VisualMapping;
            color?: VisualMapping;
            opacity?: VisualMapping;
            decal?: VisualMapping;
            symbolSize?: VisualMapping;
            liftZ?: VisualMapping;
            colorAlpha?: VisualMapping;
            colorLightness?: VisualMapping;
            colorSaturation?: VisualMapping;
            colorHue?: VisualMapping;
        } & {
            __alphaForOpacity?: VisualMapping;
        };
    };
    controllerVisuals: {
        [x: string]: {
            symbol?: VisualMapping;
            color?: VisualMapping;
            opacity?: VisualMapping;
            decal?: VisualMapping;
            symbolSize?: VisualMapping;
            liftZ?: VisualMapping;
            colorAlpha?: VisualMapping;
            colorLightness?: VisualMapping;
            colorSaturation?: VisualMapping;
            colorHue?: VisualMapping;
        } & {
            __alphaForOpacity?: VisualMapping;
        };
    };
    textStyleModel: Model<LabelOption>;
    itemSize: number[];
    init(option: Opts, parentModel: Model, ecModel: GlobalModel): void;
    /**
     * @protected
     */
    optionUpdated(newOption: Opts, isInit?: boolean): void;
    /**
     * @protected
     */
    resetVisual(supplementVisualOption: (this: this, mappingOption: VisualMappingOption, state: string) => void): void;
    /**
     * @public
     */
    getItemSymbol(): string;
    /**
     * @protected
     * @return {Array.<number>} An array of series indices.
     */
    getTargetSeriesIndices(): number[];
    /**
     * @public
     */
    eachTargetSeries<Ctx>(callback: (this: Ctx, series: SeriesModel) => void, context?: Ctx): void;
    /**
     * @pubilc
     */
    isTargetSeries(seriesModel: SeriesModel): boolean;
    /**
     * @example
     * this.formatValueText(someVal); // format single numeric value to text.
     * this.formatValueText(someVal, true); // format single category value to text.
     * this.formatValueText([min, max]); // format numeric min-max to text.
     * this.formatValueText([this.dataBound[0], max]); // using data lower bound.
     * this.formatValueText([min, this.dataBound[1]]); // using data upper bound.
     *
     * @param value Real value, or this.dataBound[0 or 1].
     * @param isCategory Only available when value is number.
     * @param edgeSymbols Open-close symbol when value is interval.
     * @protected
     */
    formatValueText(value: number | string | number[], isCategory?: boolean, edgeSymbols?: string[]): string;
    /**
     * @protected
     */
    resetExtent(): void;
    /**
     * PENDING:
     * delete this method if no outer usage.
     *
     * Return  Concrete dimension. If null/undefined is returned, no dimension is used.
     */
    getDataDimensionIndex(data: SeriesData): DimensionIndex;
    getExtent(): [number, number];
    completeVisualOption(): void;
    resetItemSize(): void;
    isCategory(): boolean;
    /**
     * @public
     * @abstract
     */
    setSelected(selected?: any): void;
    getSelected(): any;
    /**
     * @public
     * @abstract
     */
    getValueState(value: any): VisualMapModel['stateList'][number];
    /**
     * FIXME
     * Do not publish to thirt-part-dev temporarily
     * util the interface is stable. (Should it return
     * a function but not visual meta?)
     *
     * @pubilc
     * @abstract
     * @param getColorVisual
     *        params: value, valueState
     *        return: color
     * @return {Object} visualMeta
     *        should includes {stops, outerColors}
     *        outerColor means [colorBeyondMinValue, colorBeyondMaxValue]
     */
    getVisualMeta(getColorVisual: (value: number, valueState: VisualState) => string): VisualMeta;
    static defaultOption: VisualMapOption;
}
export default VisualMapModel;
