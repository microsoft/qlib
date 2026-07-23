import type Polar from '../../coord/polar/Polar.js';
import type Cartesian2D from '../../coord/cartesian/Cartesian2D.js';
import SeriesData from '../../data/SeriesData.js';
import type { LineSeriesOption } from './LineSeries.js';
interface CoordInfo {
    dataDimsForPoint: string[];
    valueStart: number;
    valueAxisDim: string;
    baseAxisDim: string;
    stacked: boolean;
    valueDim: string;
    baseDim: string;
    baseDataOffset: number;
    stackedOverDimension: string;
}
export declare function prepareDataCoordInfo(coordSys: Cartesian2D | Polar, data: SeriesData, valueOrigin?: LineSeriesOption['areaStyle']['origin']): CoordInfo;
export declare function getStackedOnPoint(dataCoordInfo: CoordInfo, coordSys: Cartesian2D | Polar, data: SeriesData, idx: number): number[];
export {};
