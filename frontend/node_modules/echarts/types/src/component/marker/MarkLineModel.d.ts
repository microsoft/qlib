import MarkerModel, { MarkerOption, MarkerStatisticType, MarkerPositionOption } from './MarkerModel.js';
import GlobalModel from '../../model/Global.js';
import { LineStyleOption, SeriesLineLabelOption, SymbolOptionMixin, ItemStyleOption, StatesOptionMixin, StatesMixinBase } from '../../util/types.js';
interface MarkLineStateOption {
    lineStyle?: LineStyleOption;
    /**
     * itemStyle for symbol
     */
    itemStyle?: ItemStyleOption;
    label?: SeriesLineLabelOption;
}
interface MarkLineDataItemOptionBase extends MarkLineStateOption, StatesOptionMixin<MarkLineStateOption, StatesMixinBase> {
    name?: string;
}
export interface MarkLine1DDataItemOption extends MarkLineDataItemOptionBase {
    xAxis?: number | string;
    yAxis?: number | string;
    type?: MarkerStatisticType;
    /**
     * When using statistic method with type.
     * valueIndex and valueDim can be specify which dim the statistic is used on.
     */
    valueIndex?: number;
    valueDim?: string;
    /**
     * Symbol for both two ends
     */
    symbol?: string[] | string;
    symbolSize?: number[] | number;
    symbolRotate?: number[] | number;
    symbolOffset?: number | string | (number | string)[];
}
interface MarkLine2DDataItemDimOption extends MarkLineDataItemOptionBase, SymbolOptionMixin, MarkerPositionOption {
}
export declare type MarkLine2DDataItemOption = [
    MarkLine2DDataItemDimOption,
    MarkLine2DDataItemDimOption
];
export interface MarkLineOption extends MarkerOption, MarkLineStateOption, StatesOptionMixin<MarkLineStateOption, StatesMixinBase> {
    mainType?: 'markLine';
    symbol?: string[] | string;
    symbolSize?: number[] | number;
    symbolRotate?: number[] | number;
    symbolOffset?: number | string | (number | string)[] | (number | string)[][];
    /**
     * Precision used on statistic method
     */
    precision?: number;
    data?: (MarkLine1DDataItemOption | MarkLine2DDataItemOption)[];
}
declare class MarkLineModel extends MarkerModel<MarkLineOption> {
    static type: string;
    type: string;
    createMarkerModelFromSeries(markerOpt: MarkLineOption, masterMarkerModel: MarkLineModel, ecModel: GlobalModel): MarkLineModel;
    static defaultOption: MarkLineOption;
}
export default MarkLineModel;
