import { Payload } from '../../util/types.js';
import GlobalModel from '../../model/Global.js';
import DataZoomModel from './DataZoomModel.js';
import { HashMap } from 'zrender/lib/core/util.js';
import SeriesModel from '../../model/Series.js';
import { CoordinateSystemHostModel } from '../../coord/CoordinateSystem.js';
import { AxisBaseModel } from '../../coord/AxisBaseModel.js';
export interface DataZoomPayloadBatchItem {
    dataZoomId: string;
    start?: number;
    end?: number;
    startValue?: number;
    endValue?: number;
}
export interface DataZoomReferCoordSysInfo {
    model: CoordinateSystemHostModel;
    axisModels: AxisBaseModel[];
}
export declare const DATA_ZOOM_AXIS_DIMENSIONS: readonly ["x", "y", "radius", "angle", "single"];
export declare type DataZoomAxisDimension = 'x' | 'y' | 'radius' | 'angle' | 'single';
declare type DataZoomAxisMainType = 'xAxis' | 'yAxis' | 'radiusAxis' | 'angleAxis' | 'singleAxis';
declare type DataZoomAxisIndexPropName = 'xAxisIndex' | 'yAxisIndex' | 'radiusAxisIndex' | 'angleAxisIndex' | 'singleAxisIndex';
declare type DataZoomAxisIdPropName = 'xAxisId' | 'yAxisId' | 'radiusAxisId' | 'angleAxisId' | 'singleAxisId';
export declare type DataZoomCoordSysMainType = 'polar' | 'grid' | 'singleAxis';
export declare function isCoordSupported(seriesModel: SeriesModel): boolean;
export declare function getAxisMainType(axisDim: DataZoomAxisDimension): DataZoomAxisMainType;
export declare function getAxisIndexPropName(axisDim: DataZoomAxisDimension): DataZoomAxisIndexPropName;
export declare function getAxisIdPropName(axisDim: DataZoomAxisDimension): DataZoomAxisIdPropName;
/**
 * If two dataZoomModels has the same axis controlled, we say that they are 'linked'.
 * This function finds all linked dataZoomModels start from the given payload.
 */
export declare function findEffectedDataZooms(ecModel: GlobalModel, payload: Payload): DataZoomModel[];
/**
 * Find the first target coordinate system.
 * Available after model built.
 *
 * @return Like {
 *                  grid: [
 *                      {model: coord0, axisModels: [axis1, axis3], coordIndex: 1},
 *                      {model: coord1, axisModels: [axis0, axis2], coordIndex: 0},
 *                      ...
 *                  ],  // cartesians must not be null/undefined.
 *                  polar: [
 *                      {model: coord0, axisModels: [axis4], coordIndex: 0},
 *                      ...
 *                  ],  // polars must not be null/undefined.
 *                  singleAxis: [
 *                      {model: coord0, axisModels: [], coordIndex: 0}
 *                  ]
 *              }
 */
export declare function collectReferCoordSysModelInfo(dataZoomModel: DataZoomModel): {
    infoList: DataZoomReferCoordSysInfo[];
    infoMap: HashMap<DataZoomReferCoordSysInfo, string>;
};
export {};
