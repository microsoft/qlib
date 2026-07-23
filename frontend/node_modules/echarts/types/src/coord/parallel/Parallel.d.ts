import * as matrix from 'zrender/lib/core/matrix.js';
import ParallelAxis from './ParallelAxis.js';
import * as graphic from '../../util/graphic.js';
import ParallelModel from './ParallelModel.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { DimensionName, ScaleDataValue } from '../../util/types.js';
import { CoordinateSystem, CoordinateSystemMaster } from '../CoordinateSystem.js';
import { ParallelActiveState } from './AxisModel.js';
import SeriesData from '../../data/SeriesData.js';
export interface ParallelAxisLayoutInfo {
    position: number[];
    rotation: number;
    transform: matrix.MatrixArray;
    axisNameAvailableWidth: number;
    axisLabelShow: boolean;
    nameTruncateMaxWidth: number;
    tickDirection: -1 | 1;
    labelDirection: -1 | 1;
}
declare type SlidedAxisExpandBehavior = 'none' | 'slide' | 'jump';
declare class Parallel implements CoordinateSystemMaster, CoordinateSystem {
    readonly type = "parallel";
    /**
     * key: dimension
     */
    private _axesMap;
    /**
     * key: dimension
     * value: {position: [], rotation, }
     */
    private _axesLayout;
    /**
     * Always follow axis order.
     */
    readonly dimensions: ParallelModel['dimensions'];
    private _rect;
    private _model;
    name: string;
    model: ParallelModel;
    constructor(parallelModel: ParallelModel, ecModel: GlobalModel, api: ExtensionAPI);
    private _init;
    /**
     * Update axis scale after data processed
     */
    update(ecModel: GlobalModel, api: ExtensionAPI): void;
    containPoint(point: number[]): boolean;
    getModel(): ParallelModel;
    /**
     * Update properties from series
     */
    private _updateAxesFromSeries;
    /**
     * Resize the parallel coordinate system.
     */
    resize(parallelModel: ParallelModel, api: ExtensionAPI): void;
    getRect(): graphic.BoundingRect;
    private _makeLayoutInfo;
    private _layoutAxes;
    /**
     * Get axis by dim.
     */
    getAxis(dim: DimensionName): ParallelAxis;
    /**
     * Convert a dim value of a single item of series data to Point.
     */
    dataToPoint(value: ScaleDataValue, dim: DimensionName): number[];
    /**
     * Travel data for one time, get activeState of each data item.
     * @param start the start dataIndex that travel from.
     * @param end the next dataIndex of the last dataIndex will be travel.
     */
    eachActiveState(data: SeriesData, callback: (activeState: ParallelActiveState, dataIndex: number) => void, start?: number, end?: number): void;
    /**
     * Whether has any activeSet.
     */
    hasAxisBrushed(): boolean;
    /**
     * Convert coords of each axis to Point.
     *  Return point. For example: [10, 20]
     */
    axisCoordToPoint(coord: number, dim: DimensionName): number[];
    /**
     * Get axis layout.
     */
    getAxisLayout(dim: DimensionName): ParallelAxisLayoutInfo;
    /**
     * @return {Object} {axisExpandWindow, delta, behavior: 'jump' | 'slide' | 'none'}.
     */
    getSlidedAxisExpandWindow(point: number[]): {
        axisExpandWindow: number[];
        behavior: SlidedAxisExpandBehavior;
    };
}
export default Parallel;
