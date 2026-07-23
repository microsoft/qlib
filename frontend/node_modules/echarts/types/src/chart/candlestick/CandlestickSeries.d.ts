import SeriesModel from '../../model/Series.js';
import { SeriesOption, SeriesOnCartesianOptionMixin, LayoutOrient, ItemStyleOption, ZRColor, ColorString, SeriesLabelOption, SeriesLargeOptionMixin, StatesOptionMixin, SeriesEncodeOptionMixin, DefaultEmphasisFocus, OptionDataValue } from '../../util/types.js';
import SeriesData from '../../data/SeriesData.js';
import Cartesian2D from '../../coord/cartesian/Cartesian2D.js';
import { BrushCommonSelectorsForSeries } from '../../component/brush/selector.js';
declare type CandlestickDataValue = OptionDataValue[];
interface CandlestickItemStyleOption extends ItemStyleOption {
    color0?: ZRColor;
    borderColor0?: ColorString;
    borderColorDoji?: ZRColor;
}
export interface CandlestickStateOption {
    itemStyle?: CandlestickItemStyleOption;
    label?: SeriesLabelOption;
}
export interface CandlestickDataItemOption extends CandlestickStateOption, StatesOptionMixin<CandlestickStateOption, ExtraStateOption> {
    value: CandlestickDataValue;
}
interface ExtraStateOption {
    emphasis?: {
        focus?: DefaultEmphasisFocus;
        scale?: boolean;
    };
}
export interface CandlestickSeriesOption extends SeriesOption<CandlestickStateOption, ExtraStateOption>, CandlestickStateOption, SeriesOnCartesianOptionMixin, SeriesLargeOptionMixin, SeriesEncodeOptionMixin {
    type?: 'candlestick';
    coordinateSystem?: 'cartesian2d';
    layout?: LayoutOrient;
    clip?: boolean;
    barMaxWidth?: number | string;
    barMinWidth?: number | string;
    barWidth?: number | string;
    data?: (CandlestickDataValue | CandlestickDataItemOption)[];
}
declare class CandlestickSeriesModel extends SeriesModel<CandlestickSeriesOption> {
    static readonly type = "series.candlestick";
    readonly type = "series.candlestick";
    static readonly dependencies: string[];
    coordinateSystem: Cartesian2D;
    dimensions: string[];
    defaultValueDimensions: {
        name: string;
        defaultTooltip: boolean;
    }[];
    static defaultOption: CandlestickSeriesOption;
    /**
     * Get dimension for shadow in dataZoom
     * @return dimension name
     */
    getShadowDim(): string;
    brushSelector(dataIndex: number, data: SeriesData, selectors: BrushCommonSelectorsForSeries): boolean;
}
export default CandlestickSeriesModel;
