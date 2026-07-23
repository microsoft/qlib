import SeriesModel from '../../model/Series.js';
import { SeriesOption, SeriesOnCartesianOptionMixin, SeriesOnPolarOptionMixin, SeriesOnCalendarOptionMixin, SeriesOnGeoOptionMixin, SeriesOnSingleOptionMixin, OptionDataValue, ItemStyleOption, SeriesLabelOption, SeriesLargeOptionMixin, SeriesStackOptionMixin, SymbolOptionMixin, StatesOptionMixin, OptionDataItemObject, SeriesEncodeOptionMixin, CallbackDataParams, DefaultEmphasisFocus } from '../../util/types.js';
import GlobalModel from '../../model/Global.js';
import SeriesData from '../../data/SeriesData.js';
import { BrushCommonSelectorsForSeries } from '../../component/brush/selector.js';
interface ScatterStateOption<TCbParams = never> {
    itemStyle?: ItemStyleOption<TCbParams>;
    label?: SeriesLabelOption;
}
interface ScatterStatesOptionMixin {
    emphasis?: {
        focus?: DefaultEmphasisFocus;
        scale?: boolean | number;
    };
}
export interface ScatterDataItemOption extends SymbolOptionMixin, ScatterStateOption, StatesOptionMixin<ScatterStateOption, ScatterStatesOptionMixin>, OptionDataItemObject<OptionDataValue> {
}
export interface ScatterSeriesOption extends SeriesOption<ScatterStateOption<CallbackDataParams>, ScatterStatesOptionMixin>, ScatterStateOption<CallbackDataParams>, SeriesOnCartesianOptionMixin, SeriesOnPolarOptionMixin, SeriesOnCalendarOptionMixin, SeriesOnGeoOptionMixin, SeriesOnSingleOptionMixin, SeriesLargeOptionMixin, SeriesStackOptionMixin, SymbolOptionMixin<CallbackDataParams>, SeriesEncodeOptionMixin {
    type?: 'scatter';
    coordinateSystem?: string;
    cursor?: string;
    clip?: boolean;
    data?: (ScatterDataItemOption | OptionDataValue | OptionDataValue[])[] | ArrayLike<number>;
}
declare class ScatterSeriesModel extends SeriesModel<ScatterSeriesOption> {
    static readonly type = "series.scatter";
    type: string;
    static readonly dependencies: string[];
    hasSymbolVisual: boolean;
    getInitialData(option: ScatterSeriesOption, ecModel: GlobalModel): SeriesData;
    getProgressive(): number | false;
    getProgressiveThreshold(): number;
    brushSelector(dataIndex: number, data: SeriesData, selectors: BrushCommonSelectorsForSeries): boolean;
    getZLevelKey(): string;
    static defaultOption: ScatterSeriesOption;
}
export default ScatterSeriesModel;
