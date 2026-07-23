import Scale from '../scale/Scale.js';
import { AxisBaseModel } from './AxisBaseModel.js';
import { ScaleDataValue } from '../util/types.js';
export interface ScaleRawExtentResult {
    readonly min: number;
    readonly max: number;
    readonly minFixed: boolean;
    readonly maxFixed: boolean;
    readonly isBlank: boolean;
}
export declare class ScaleRawExtentInfo {
    private _needCrossZero;
    private _isOrdinal;
    private _axisDataLen;
    private _boundaryGapInner;
    private _modelMinRaw;
    private _modelMaxRaw;
    private _modelMinNum;
    private _modelMaxNum;
    private _dataMin;
    private _dataMax;
    private _determinedMin;
    private _determinedMax;
    readonly frozen: boolean;
    constructor(scale: Scale, model: AxisBaseModel, originalExtent: number[]);
    /**
     * Parameters depending on outside (like model, user callback)
     * are prepared and fixed here.
     */
    private _prepareParams;
    /**
     * Calculate extent by prepared parameters.
     * This method has no external dependency and can be called duplicatedly,
     * getting the same result.
     * If parameters changed, should call this method to recalcuate.
     */
    calculate(): ScaleRawExtentResult;
    modifyDataMinMax(minMaxName: 'min' | 'max', val: number): void;
    setDeterminedMinMax(minMaxName: 'min' | 'max', val: number): void;
    freeze(): void;
}
/**
 * Get scale min max and related info only depends on model settings.
 * This method can be called after coordinate system created.
 * For example, in data processing stage.
 *
 * Scale extent info probably be required multiple times during a workflow.
 * For example:
 * (1) `dataZoom` depends it to get the axis extent in "100%" state.
 * (2) `processor/extentCalculator` depends it to make sure whether axis extent is specified.
 * (3) `coordSys.update` use it to finally decide the scale extent.
 * But the callback of `min`/`max` should not be called multiple times.
 * The code below should not be implemented repeatedly either.
 * So we cache the result in the scale instance, which will be recreated at the beginning
 * of the workflow (because `scale` instance will be recreated each round of the workflow).
 */
export declare function ensureScaleRawExtentInfo(scale: Scale, model: AxisBaseModel, originalExtent: number[]): ScaleRawExtentInfo;
export declare function parseAxisModelMinMax(scale: Scale, minMax: ScaleDataValue): number;
