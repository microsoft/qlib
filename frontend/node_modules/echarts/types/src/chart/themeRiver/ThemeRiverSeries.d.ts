import SeriesModel from '../../model/Series.js';
import SeriesData from '../../data/SeriesData.js';
import * as zrUtil from 'zrender/lib/core/util.js';
import { SeriesOption, SeriesOnSingleOptionMixin, OptionDataValueDate, OptionDataValueNumeric, ItemStyleOption, BoxLayoutOptionMixin, ZRColor, SeriesLabelOption, CallbackDataParams, DefaultStatesMixinEmphasis } from '../../util/types.js';
import SingleAxis from '../../coord/single/SingleAxis.js';
import GlobalModel from '../../model/Global.js';
import Single from '../../coord/single/Single.js';
interface ThemeRiverSeriesLabelOption extends SeriesLabelOption {
    margin?: number;
}
declare type ThemerRiverDataItem = [OptionDataValueDate, OptionDataValueNumeric, string];
interface ThemeRiverStatesMixin {
    emphasis?: DefaultStatesMixinEmphasis;
}
export interface ThemeRiverStateOption<TCbParams = never> {
    label?: ThemeRiverSeriesLabelOption;
    itemStyle?: ItemStyleOption<TCbParams>;
}
export interface ThemeRiverSeriesOption extends SeriesOption<ThemeRiverStateOption<CallbackDataParams>, ThemeRiverStatesMixin>, ThemeRiverStateOption<CallbackDataParams>, SeriesOnSingleOptionMixin, BoxLayoutOptionMixin {
    type?: 'themeRiver';
    color?: ZRColor[];
    coordinateSystem?: 'singleAxis';
    /**
     * gap in axis's orthogonal orientation
     */
    boundaryGap?: (string | number)[];
    /**
     * [date, value, name]
     */
    data?: ThemerRiverDataItem[];
}
declare class ThemeRiverSeriesModel extends SeriesModel<ThemeRiverSeriesOption> {
    static readonly type = "series.themeRiver";
    readonly type = "series.themeRiver";
    static readonly dependencies: string[];
    nameMap: zrUtil.HashMap<number, string>;
    coordinateSystem: Single;
    /**
     * @override
     */
    init(option: ThemeRiverSeriesOption): void;
    /**
     * If there is no value of a certain point in the time for some event,set it value to 0.
     *
     * @param {Array} data  initial data in the option
     * @return {Array}
     */
    fixData(data: ThemeRiverSeriesOption['data']): ThemerRiverDataItem[];
    /**
     * @override
     * @param  option  the initial option that user gave
     * @param  ecModel  the model object for themeRiver option
     */
    getInitialData(option: ThemeRiverSeriesOption, ecModel: GlobalModel): SeriesData;
    /**
     * The raw data is divided into multiple layers and each layer
     *     has same name.
     */
    getLayerSeries(): {
        name: string;
        indices: number[];
    }[];
    /**
     * Get data indices for show tooltip content
     */
    getAxisTooltipData(dim: string | string[], value: number, baseAxis: SingleAxis): {
        dataIndices: number[];
        nestestValue: number;
    };
    formatTooltip(dataIndex: number, multipleSeries: boolean, dataType: string): import("../../component/tooltip/tooltipMarkup").TooltipMarkupNameValueBlock;
    static defaultOption: ThemeRiverSeriesOption;
}
export default ThemeRiverSeriesModel;
