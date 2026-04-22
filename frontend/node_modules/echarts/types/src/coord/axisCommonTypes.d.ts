import { TextAlign, TextVerticalAlign } from 'zrender/lib/core/types.js';
import { TextCommonOption, LineStyleOption, OrdinalRawValue, ZRColor, AreaStyleOption, ComponentOption, ColorString, AnimationOptionMixin, Dictionary, ScaleDataValue, CommonAxisPointerOption } from '../util/types.js';
import { TextStyleProps } from 'zrender/lib/graphic/Text.js';
export declare const AXIS_TYPES: {
    readonly value: 1;
    readonly category: 1;
    readonly time: 1;
    readonly log: 1;
};
export declare type OptionAxisType = keyof typeof AXIS_TYPES;
export interface AxisBaseOptionCommon extends ComponentOption, AnimationOptionMixin {
    type?: OptionAxisType;
    show?: boolean;
    inverse?: boolean;
    name?: string;
    nameLocation?: 'start' | 'middle' | 'end';
    nameRotate?: number;
    nameTruncate?: {
        maxWidth?: number;
        ellipsis?: string;
        placeholder?: string;
    };
    nameTextStyle?: AxisNameTextStyleOption;
    nameGap?: number;
    silent?: boolean;
    triggerEvent?: boolean;
    tooltip?: {
        show?: boolean;
    };
    axisLabel?: AxisLabelBaseOption;
    axisPointer?: CommonAxisPointerOption;
    axisLine?: AxisLineOption;
    axisTick?: AxisTickOption;
    minorTick?: MinorTickOption;
    splitLine?: SplitLineOption;
    minorSplitLine?: MinorSplitLineOption;
    splitArea?: SplitAreaOption;
    /**
     * Min value of the axis. can be:
     * + ScaleDataValue
     * + 'dataMin': use the min value in data.
     * + null/undefined: auto decide min value (consider pretty look and boundaryGap).
     */
    min?: ScaleDataValue | 'dataMin' | ((extent: {
        min: number;
        max: number;
    }) => ScaleDataValue);
    /**
     * Max value of the axis. can be:
     * + ScaleDataValue
     * + 'dataMax': use the max value in data.
     * + null/undefined: auto decide max value (consider pretty look and boundaryGap).
     */
    max?: ScaleDataValue | 'dataMax' | ((extent: {
        min: number;
        max: number;
    }) => ScaleDataValue);
    startValue?: number;
}
export interface NumericAxisBaseOptionCommon extends AxisBaseOptionCommon {
    boundaryGap?: [number | string, number | string];
    /**
     * AxisTick and axisLabel and splitLine are calculated based on splitNumber.
     */
    splitNumber?: number;
    /**
     * Interval specifies the span of the ticks is mandatorily.
     */
    interval?: number;
    /**
     * Specify min interval when auto calculate tick interval.
     */
    minInterval?: number;
    /**
     * Specify max interval when auto calculate tick interval.
     */
    maxInterval?: number;
    /**
     * If align ticks to the first axis that is not use alignTicks
     * If all axes has alignTicks: true. The first one will be applied.
     *
     * Will be ignored if interval is set.
     */
    alignTicks?: boolean;
}
export interface CategoryAxisBaseOption extends AxisBaseOptionCommon {
    type?: 'category';
    boundaryGap?: boolean;
    axisLabel?: AxisLabelOption<'category'> & {
        interval?: 'auto' | number | ((index: number, value: string) => boolean);
    };
    data?: (OrdinalRawValue | {
        value: OrdinalRawValue;
        textStyle?: TextCommonOption;
    })[];
    deduplication?: boolean;
    axisTick?: AxisBaseOptionCommon['axisTick'] & {
        alignWithLabel?: boolean;
        interval?: 'auto' | number | ((index: number, value: string) => boolean);
    };
}
export interface ValueAxisBaseOption extends NumericAxisBaseOptionCommon {
    type?: 'value';
    axisLabel?: AxisLabelOption<'value'>;
    /**
     * Optional value can be:
     * + `false`: always include value 0.
     * + `true`: the axis may not contain zero position.
     */
    scale?: boolean;
}
export interface LogAxisBaseOption extends NumericAxisBaseOptionCommon {
    type?: 'log';
    axisLabel?: AxisLabelOption<'log'>;
    logBase?: number;
}
export interface TimeAxisBaseOption extends NumericAxisBaseOptionCommon {
    type?: 'time';
    axisLabel?: AxisLabelOption<'time'>;
}
interface AxisNameTextStyleOption extends TextCommonOption {
    rich?: Dictionary<TextCommonOption>;
}
interface AxisLineOption {
    show?: boolean | 'auto';
    onZero?: boolean;
    onZeroAxisIndex?: number;
    symbol?: string | [string, string];
    symbolSize?: number[];
    symbolOffset?: string | number | (string | number)[];
    lineStyle?: LineStyleOption;
}
interface AxisTickOption {
    show?: boolean | 'auto';
    inside?: boolean;
    length?: number;
    lineStyle?: LineStyleOption;
    customValues?: (number | string | Date)[];
}
declare type AxisLabelValueFormatter = (value: number, index: number) => string;
declare type AxisLabelCategoryFormatter = (value: string, index: number) => string;
declare type TimeAxisLabelUnitFormatter = AxisLabelValueFormatter | string[] | string;
export declare type TimeAxisLabelFormatterOption = string | ((value: number, index: number, extra: {
    level: number;
}) => string) | {
    year?: TimeAxisLabelUnitFormatter;
    month?: TimeAxisLabelUnitFormatter;
    week?: TimeAxisLabelUnitFormatter;
    day?: TimeAxisLabelUnitFormatter;
    hour?: TimeAxisLabelUnitFormatter;
    minute?: TimeAxisLabelUnitFormatter;
    second?: TimeAxisLabelUnitFormatter;
    millisecond?: TimeAxisLabelUnitFormatter;
    inherit?: boolean;
};
declare type LabelFormatters = {
    value: AxisLabelValueFormatter | string;
    log: AxisLabelValueFormatter | string;
    category: AxisLabelCategoryFormatter | string;
    time: TimeAxisLabelFormatterOption;
};
interface AxisLabelBaseOption extends Omit<TextCommonOption, 'color'> {
    show?: boolean;
    inside?: boolean;
    rotate?: number;
    showMinLabel?: boolean;
    showMaxLabel?: boolean;
    alignMinLabel?: TextAlign;
    alignMaxLabel?: TextAlign;
    verticalAlignMinLabel?: TextVerticalAlign;
    verticalAlignMaxLabel?: TextVerticalAlign;
    margin?: number;
    rich?: Dictionary<TextCommonOption>;
    /**
     * If hide overlapping labels.
     */
    hideOverlap?: boolean;
    customValues?: (number | string | Date)[];
    color?: ColorString | ((value?: string | number, index?: number) => ColorString);
    overflow?: TextStyleProps['overflow'];
}
interface AxisLabelOption<TType extends OptionAxisType> extends AxisLabelBaseOption {
    formatter?: LabelFormatters[TType];
}
interface MinorTickOption {
    show?: boolean;
    splitNumber?: number;
    length?: number;
    lineStyle?: LineStyleOption;
}
interface SplitLineOption {
    show?: boolean;
    interval?: 'auto' | number | ((index: number, value: string) => boolean);
    showMinLine?: boolean;
    showMaxLine?: boolean;
    lineStyle?: LineStyleOption<ZRColor | ZRColor[]>;
}
interface MinorSplitLineOption {
    show?: boolean;
    lineStyle?: LineStyleOption;
}
interface SplitAreaOption {
    show?: boolean;
    interval?: 'auto' | number | ((index: number, value: string) => boolean);
    areaStyle?: AreaStyleOption<ZRColor[]>;
}
export declare type AxisBaseOption = ValueAxisBaseOption | LogAxisBaseOption | CategoryAxisBaseOption | TimeAxisBaseOption | AxisBaseOptionCommon;
export {};
