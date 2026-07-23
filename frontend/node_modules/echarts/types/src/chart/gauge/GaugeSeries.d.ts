import SeriesModel from '../../model/Series.js';
import { SeriesOption, CircleLayoutOptionMixin, LineStyleOption, ColorString, LabelOption, ItemStyleOption, OptionDataValueNumeric, StatesOptionMixin, SeriesEncodeOptionMixin, DefaultStatesMixinEmphasis, CallbackDataParams } from '../../util/types.js';
import GlobalModel from '../../model/Global.js';
import SeriesData from '../../data/SeriesData.js';
declare type GaugeColorStop = [number, ColorString];
interface LabelFormatter {
    (value: number): string;
}
interface PointerOption {
    icon?: string;
    show?: boolean;
    /**
     * If pointer shows above title and detail
     */
    showAbove?: boolean;
    keepAspect?: boolean;
    itemStyle?: ItemStyleOption;
    /**
     * Can be percent
     */
    offsetCenter?: (number | string)[];
    length?: number | string;
    width?: number;
}
interface AnchorOption {
    show?: boolean;
    showAbove?: boolean;
    size?: number;
    icon?: string;
    offsetCenter?: (number | string)[];
    keepAspect?: boolean;
    itemStyle?: ItemStyleOption;
}
interface ProgressOption {
    show?: boolean;
    overlap?: boolean;
    width?: number;
    roundCap?: boolean;
    clip?: boolean;
    itemStyle?: ItemStyleOption;
}
interface TitleOption extends LabelOption {
    /**
     * [x, y] offset
     */
    offsetCenter?: (number | string)[];
    formatter?: LabelFormatter | string;
    /**
     * If do value animtion.
     */
    valueAnimation?: boolean;
}
interface DetailOption extends LabelOption {
    /**
     * [x, y] offset
     */
    offsetCenter?: (number | string)[];
    formatter?: LabelFormatter | string;
    /**
     * If do value animtion.
     */
    valueAnimation?: boolean;
}
interface GaugeStatesMixin {
    emphasis?: DefaultStatesMixinEmphasis;
}
export interface GaugeStateOption<TCbParams = never> {
    itemStyle?: ItemStyleOption<TCbParams>;
}
export interface GaugeDataItemOption extends GaugeStateOption, StatesOptionMixin<GaugeStateOption<CallbackDataParams>, GaugeStatesMixin> {
    name?: string;
    value?: OptionDataValueNumeric;
    pointer?: PointerOption;
    progress?: ProgressOption;
    title?: TitleOption;
    detail?: DetailOption;
}
export interface GaugeSeriesOption extends SeriesOption<GaugeStateOption, GaugeStatesMixin>, GaugeStateOption<CallbackDataParams>, CircleLayoutOptionMixin, SeriesEncodeOptionMixin {
    type?: 'gauge';
    radius?: number | string;
    startAngle?: number;
    endAngle?: number;
    clockwise?: boolean;
    min?: number;
    max?: number;
    splitNumber?: number;
    itemStyle?: ItemStyleOption;
    axisLine?: {
        show?: boolean;
        roundCap?: boolean;
        lineStyle?: Omit<LineStyleOption, 'color'> & {
            color?: GaugeColorStop[];
        };
    };
    progress?: ProgressOption;
    splitLine?: {
        show?: boolean;
        /**
         * Can be percent
         */
        length?: number;
        distance?: number;
        lineStyle?: LineStyleOption;
    };
    axisTick?: {
        show?: boolean;
        splitNumber?: number;
        /**
         * Can be percent
         */
        length?: number | string;
        distance?: number;
        lineStyle?: LineStyleOption;
    };
    axisLabel?: Omit<LabelOption, 'rotate'> & {
        formatter?: LabelFormatter | string;
        rotate?: 'tangential' | 'radial' | number;
    };
    pointer?: PointerOption;
    anchor?: AnchorOption;
    title?: TitleOption;
    detail?: DetailOption;
    data?: (OptionDataValueNumeric | GaugeDataItemOption)[];
}
declare class GaugeSeriesModel extends SeriesModel<GaugeSeriesOption> {
    static type: "series.gauge";
    type: "series.gauge";
    visualStyleAccessPath: string;
    getInitialData(option: GaugeSeriesOption, ecModel: GlobalModel): SeriesData;
    static defaultOption: GaugeSeriesOption;
}
export default GaugeSeriesModel;
