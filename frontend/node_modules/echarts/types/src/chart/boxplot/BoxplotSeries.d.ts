import SeriesModel from '../../model/Series.js';
import { WhiskerBoxCommonMixin } from '../helper/whiskerBoxCommon.js';
import { SeriesOption, SeriesOnCartesianOptionMixin, LayoutOrient, ItemStyleOption, SeriesLabelOption, OptionDataValueNumeric, StatesOptionMixin, SeriesEncodeOptionMixin, DefaultEmphasisFocus, CallbackDataParams } from '../../util/types.js';
import type Axis2D from '../../coord/cartesian/Axis2D.js';
import Cartesian2D from '../../coord/cartesian/Cartesian2D.js';
declare type BoxplotDataValue = OptionDataValueNumeric[];
export interface BoxplotStateOption<TCbParams = never> {
    itemStyle?: ItemStyleOption<TCbParams>;
    label?: SeriesLabelOption;
}
export interface BoxplotDataItemOption extends BoxplotStateOption, StatesOptionMixin<BoxplotStateOption, ExtraStateOption> {
    value: BoxplotDataValue;
}
interface ExtraStateOption {
    emphasis?: {
        focus?: DefaultEmphasisFocus;
        scale?: boolean;
    };
}
export interface BoxplotSeriesOption extends SeriesOption<BoxplotStateOption<CallbackDataParams>, ExtraStateOption>, BoxplotStateOption<CallbackDataParams>, SeriesOnCartesianOptionMixin, SeriesEncodeOptionMixin {
    type?: 'boxplot';
    coordinateSystem?: 'cartesian2d';
    layout?: LayoutOrient;
    /**
     * [min, max] can be percent of band width.
     */
    boxWidth?: (string | number)[];
    data?: (BoxplotDataValue | BoxplotDataItemOption)[];
}
declare class BoxplotSeriesModel extends SeriesModel<BoxplotSeriesOption> {
    static readonly type = "series.boxplot";
    readonly type = "series.boxplot";
    static readonly dependencies: string[];
    coordinateSystem: Cartesian2D;
    /**
     * @see <https://en.wikipedia.org/wiki/Box_plot>
     * The meanings of 'min' and 'max' depend on user,
     * and echarts do not need to know it.
     * @readOnly
     */
    defaultValueDimensions: {
        name: string;
        defaultTooltip: boolean;
    }[];
    dimensions: string[];
    visualDrawType: "stroke";
    static defaultOption: BoxplotSeriesOption;
}
interface BoxplotSeriesModel extends WhiskerBoxCommonMixin<BoxplotSeriesOption> {
    getBaseAxis(): Axis2D;
}
export default BoxplotSeriesModel;
