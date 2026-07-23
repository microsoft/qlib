import { DataFormatMixin } from '../../model/mixin/dataFormat.js';
import ComponentModel from '../../model/Component.js';
import SeriesModel from '../../model/Series.js';
import { ComponentOption, AnimationOptionMixin, CommonTooltipOption, ScaleDataValue, CallbackDataParams, SeriesDataType } from '../../util/types.js';
import Model from '../../model/Model.js';
import GlobalModel from '../../model/Global.js';
import SeriesData from '../../data/SeriesData.js';
export declare type MarkerStatisticType = 'average' | 'min' | 'max' | 'median';
/**
 * Option to specify where to put the marker.
 */
export interface MarkerPositionOption {
    x?: number | string;
    y?: number | string;
    /**
     * Coord on any coordinate system
     */
    coord?: (ScaleDataValue | MarkerStatisticType)[];
    xAxis?: ScaleDataValue;
    yAxis?: ScaleDataValue;
    radiusAxis?: ScaleDataValue;
    angleAxis?: ScaleDataValue;
    type?: MarkerStatisticType;
    /**
     * When using statistic method with type.
     * valueIndex and valueDim can be specify which dim the statistic is used on.
     */
    valueIndex?: number;
    valueDim?: string;
    /**
     * Value to be displayed as label. Totally optional
     */
    value?: string | number;
}
export interface MarkerOption extends ComponentOption, AnimationOptionMixin {
    silent?: boolean;
    data?: unknown[];
    tooltip?: CommonTooltipOption<unknown> & {
        trigger?: 'item' | 'axis' | boolean | 'none';
    };
}
declare abstract class MarkerModel<Opts extends MarkerOption = MarkerOption> extends ComponentModel<Opts> {
    static type: string;
    type: string;
    /**
     * If marker model is created by self from series
     */
    createdBySelf: boolean;
    static readonly dependencies: string[];
    __hostSeries: SeriesModel;
    private _data;
    /**
     * @overrite
     */
    init(option: Opts, parentModel: Model, ecModel: GlobalModel): void;
    isAnimationEnabled(): boolean;
    /**
     * @overrite
     */
    mergeOption(newOpt: Opts, ecModel: GlobalModel): void;
    _mergeOption(newOpt: Opts, ecModel: GlobalModel, createdBySelf?: boolean, isInit?: boolean): void;
    formatTooltip(dataIndex: number, multipleSeries: boolean, dataType: string): import("../tooltip/tooltipMarkup").TooltipMarkupSection;
    getData(): SeriesData<this>;
    setData(data: SeriesData): void;
    getDataParams(dataIndex: number, dataType?: SeriesDataType): CallbackDataParams;
    /**
     * Create slave marker model from series.
     */
    abstract createMarkerModelFromSeries(markerOpt: Opts, masterMarkerModel: MarkerModel, ecModel: GlobalModel): MarkerModel;
    static getMarkerModelFromSeries(seriesModel: SeriesModel, componentType: 'markLine' | 'markPoint' | 'markArea'): MarkerModel<MarkerOption>;
}
interface MarkerModel<Opts extends MarkerOption = MarkerOption> extends DataFormatMixin {
}
export default MarkerModel;
