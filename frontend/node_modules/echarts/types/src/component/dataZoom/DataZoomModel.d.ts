import { HashMap } from 'zrender/lib/core/util.js';
import AxisProxy from './AxisProxy.js';
import ComponentModel from '../../model/Component.js';
import { LayoutOrient, ComponentOption, LabelOption } from '../../util/types.js';
import Model from '../../model/Model.js';
import GlobalModel from '../../model/Global.js';
import { AxisBaseModel } from '../../coord/AxisBaseModel.js';
import { DataZoomAxisDimension } from './helper.js';
export interface DataZoomOption extends ComponentOption {
    mainType?: 'dataZoom';
    /**
     * Default auto by axisIndex
     */
    orient?: LayoutOrient;
    /**
     * Default the first horizontal category axis.
     */
    xAxisIndex?: number | number[];
    xAxisId?: string | string[];
    /**
     * Default the first vertical category axis.
     */
    yAxisIndex?: number | number[];
    yAxisId?: string | string[];
    radiusAxisIndex?: number | number[];
    radiusAxisId?: string | string[];
    angleAxisIndex?: number | number[];
    angleAxisId?: string | string[];
    singleAxisIndex?: number | number[];
    singleAxisId?: string | string[];
    /**
     * Possible values: 'filter' or 'empty' or 'weakFilter'.
     * 'filter': data items which are out of window will be removed. This option is
     *         applicable when filtering outliers. For each data item, it will be
     *         filtered if one of the relevant dimensions is out of the window.
     * 'weakFilter': data items which are out of window will be removed. This option
     *         is applicable when filtering outliers. For each data item, it will be
     *         filtered only if all  of the relevant dimensions are out of the same
     *         side of the window.
     * 'empty': data items which are out of window will be set to empty.
     *         This option is applicable when user should not neglect
     *         that there are some data items out of window.
     * 'none': Do not filter.
     * Taking line chart as an example, line will be broken in
     * the filtered points when filterModel is set to 'empty', but
     * be connected when set to 'filter'.
     */
    filterMode?: 'filter' | 'weakFilter' | 'empty' | 'none';
    /**
     * Dispatch action by the fixed rate, avoid frequency.
     * default 100. Do not throttle when use null/undefined.
     * If animation === true and animationDurationUpdate > 0,
     * default value is 100, otherwise 20.
     */
    throttle?: number | null | undefined;
    /**
     * Start percent. 0 ~ 100
     */
    start?: number;
    /**
     * End percent. 0 ~ 100
     */
    end?: number;
    /**
     * Start value. If startValue specified, start is ignored
     */
    startValue?: number | string | Date;
    /**
     * End value. If endValue specified, end is ignored.
     */
    endValue?: number | string | Date;
    /**
     * Min span percent, 0 - 100
     * The range of dataZoom can not be smaller than that.
     */
    minSpan?: number;
    /**
     * Max span percent, 0 - 100
     * The range of dataZoom can not be larger than that.
     */
    maxSpan?: number;
    minValueSpan?: number;
    maxValueSpan?: number;
    rangeMode?: ['value' | 'percent', 'value' | 'percent'];
    realtime?: boolean;
    textStyle?: LabelOption;
}
declare type RangeOption = Pick<DataZoomOption, 'start' | 'end' | 'startValue' | 'endValue'>;
export declare type DataZoomExtendedAxisBaseModel = AxisBaseModel & {
    __dzAxisProxy: AxisProxy;
};
declare class DataZoomAxisInfo {
    indexList: number[];
    indexMap: boolean[];
    add(axisCmptIdx: number): void;
}
export declare type DataZoomTargetAxisInfoMap = HashMap<DataZoomAxisInfo, DataZoomAxisDimension>;
declare class DataZoomModel<Opts extends DataZoomOption = DataZoomOption> extends ComponentModel<Opts> {
    static type: string;
    type: string;
    static dependencies: string[];
    static defaultOption: DataZoomOption;
    private _autoThrottle;
    private _orient;
    private _targetAxisInfoMap;
    private _noTarget;
    /**
     * It is `[rangeModeForMin, rangeModeForMax]`.
     * The optional values for `rangeMode`:
     * + `'value'` mode: the axis extent will always be determined by
     *     `dataZoom.startValue` and `dataZoom.endValue`, despite
     *     how data like and how `axis.min` and `axis.max` are.
     * + `'percent'` mode: `100` represents 100% of the `[dMin, dMax]`,
     *     where `dMin` is `axis.min` if `axis.min` specified, otherwise `data.extent[0]`,
     *     and `dMax` is `axis.max` if `axis.max` specified, otherwise `data.extent[1]`.
     *     Axis extent will be determined by the result of the percent of `[dMin, dMax]`.
     *
     * For example, when users are using dynamic data (update data periodically via `setOption`),
     * if in `'value`' mode, the window will be kept in a fixed value range despite how
     * data are appended, while if in `'percent'` mode, whe window range will be changed alone with
     * the appended data (suppose `axis.min` and `axis.max` are not specified).
     */
    private _rangePropMode;
    /**
     * @readonly
     */
    settledOption: Opts;
    init(option: Opts, parentModel: Model, ecModel: GlobalModel): void;
    mergeOption(newOption: Opts): void;
    private _doInit;
    private _resetTarget;
    private _fillSpecifiedTargetAxis;
    private _fillAutoTargetAxisByOrient;
    private _makeAutoOrientByTargetAxis;
    private _setDefaultThrottle;
    private _updateRangeUse;
    noTarget(): boolean;
    getFirstTargetAxisModel(): AxisBaseModel;
    /**
     * @param {Function} callback param: axisModel, dimNames, axisIndex, dataZoomModel, ecModel
     */
    eachTargetAxis<Ctx>(callback: (this: Ctx, axisDim: DataZoomAxisDimension, axisIndex: number) => void, context?: Ctx): void;
    /**
     * @return If not found, return null/undefined.
     */
    getAxisProxy(axisDim: DataZoomAxisDimension, axisIndex: number): AxisProxy;
    /**
     * @return If not found, return null/undefined.
     */
    getAxisModel(axisDim: DataZoomAxisDimension, axisIndex: number): AxisBaseModel;
    /**
     * If not specified, set to undefined.
     */
    setRawRange(opt: RangeOption): void;
    setCalculatedRange(opt: RangeOption): void;
    getPercentRange(): number[];
    /**
     * For example, chart.getModel().getComponent('dataZoom').getValueRange('y', 0);
     *
     * @return [startValue, endValue] value can only be '-' or finite number.
     */
    getValueRange(axisDim: DataZoomAxisDimension, axisIndex: number): number[];
    /**
     * @param axisModel If axisModel given, find axisProxy
     *      corresponding to the axisModel
     */
    findRepresentativeAxisProxy(axisModel?: AxisBaseModel): AxisProxy;
    getRangePropMode(): DataZoomModel['_rangePropMode'];
    getOrient(): LayoutOrient;
}
export default DataZoomModel;
