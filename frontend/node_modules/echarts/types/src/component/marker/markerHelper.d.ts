import SeriesModel from '../../model/Series.js';
import SeriesData from '../../data/SeriesData.js';
import { MarkerStatisticType, MarkerPositionOption } from './MarkerModel.js';
import Axis from '../../coord/Axis.js';
import { CoordinateSystem } from '../../coord/CoordinateSystem.js';
import { ScaleDataValue, ParsedValue, DimensionName } from '../../util/types.js';
import SeriesDimensionDefine from '../../data/SeriesDimensionDefine.js';
interface MarkerAxisInfo {
    valueDataDim: DimensionName;
    valueAxis: Axis;
    baseAxis: Axis;
    baseDataDim: DimensionName;
}
export declare type MarkerDimValueGetter<TMarkerItemOption> = (item: TMarkerItemOption, dimName: string, dataIndex: number, dimIndex: number) => ParsedValue;
/**
 * Transform markPoint data item to format used in List by do the following
 * 1. Calculate statistic like `max`, `min`, `average`
 * 2. Convert `item.xAxis`, `item.yAxis` to `item.coord` array
 */
export declare function dataTransform(seriesModel: SeriesModel, item: MarkerPositionOption): MarkerPositionOption;
export declare function getAxisInfo(item: MarkerPositionOption, data: SeriesData, coordSys: CoordinateSystem, seriesModel: SeriesModel): MarkerAxisInfo;
/**
 * Filter data which is out of coordinateSystem range
 * [dataFilter description]
 */
export declare function dataFilter(coordSys: CoordinateSystem & {
    containData?(data: ScaleDataValue[]): boolean;
}, item: MarkerPositionOption): boolean;
export declare function zoneFilter(coordSys: CoordinateSystem & {
    containZone?(data1: ScaleDataValue[], data2: ScaleDataValue[]): boolean;
}, item1: MarkerPositionOption, item2: MarkerPositionOption): boolean;
export declare function createMarkerDimValueGetter(inCoordSys: boolean, dims: SeriesDimensionDefine[]): MarkerDimValueGetter<MarkerPositionOption>;
export declare function numCalculate(data: SeriesData, valueDataDim: string, type: MarkerStatisticType): number;
export {};
