import GlobalModel from '../../model/Global.js';
import SeriesModel from '../../model/Series.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import DataZoomModel from './DataZoomModel.js';
import { AxisBaseModel } from '../../coord/AxisBaseModel.js';
import { DataZoomAxisDimension } from './helper.js';
interface MinMaxSpan {
    minSpan: number;
    maxSpan: number;
    minValueSpan: number;
    maxValueSpan: number;
}
/**
 * Operate single axis.
 * One axis can only operated by one axis operator.
 * Different dataZoomModels may be defined to operate the same axis.
 * (i.e. 'inside' data zoom and 'slider' data zoom components)
 * So dataZoomModels share one axisProxy in that case.
 */
declare class AxisProxy {
    ecModel: GlobalModel;
    private _dimName;
    private _axisIndex;
    private _valueWindow;
    private _percentWindow;
    private _dataExtent;
    private _minMaxSpan;
    private _dataZoomModel;
    constructor(dimName: DataZoomAxisDimension, axisIndex: number, dataZoomModel: DataZoomModel, ecModel: GlobalModel);
    /**
     * Whether the axisProxy is hosted by dataZoomModel.
     */
    hostedBy(dataZoomModel: DataZoomModel): boolean;
    /**
     * @return Value can only be NaN or finite value.
     */
    getDataValueWindow(): [number, number];
    /**
     * @return {Array.<number>}
     */
    getDataPercentWindow(): [number, number];
    getTargetSeriesModels(): SeriesModel<import("../../util/types").SeriesOption<unknown, import("../../util/types").DefaultStatesMixin>>[];
    getAxisModel(): AxisBaseModel;
    getMinMaxSpan(): MinMaxSpan;
    /**
     * Only calculate by given range and this._dataExtent, do not change anything.
     */
    calculateDataWindow(opt?: {
        start?: number;
        end?: number;
        startValue?: number | string | Date;
        endValue?: number | string | Date;
    }): {
        valueWindow: [number, number];
        percentWindow: [number, number];
    };
    /**
     * Notice: reset should not be called before series.restoreData() is called,
     * so it is recommended to be called in "process stage" but not "model init
     * stage".
     */
    reset(dataZoomModel: DataZoomModel): void;
    filterData(dataZoomModel: DataZoomModel, api: ExtensionAPI): void;
    private _updateMinMaxSpan;
    private _setAxisModel;
}
export default AxisProxy;
